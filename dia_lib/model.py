# ComfyUI-DiaTTS/dia_lib/model.py

import time
from enum import Enum

import dac
import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from .audio import apply_audio_delay, build_delay_indices, build_revert_indices, decode, revert_audio_delay
from .config import DiaConfig
from .layers import DiaModel
from .state import DecoderInferenceState, DecoderOutput, EncoderInferenceState

DEFAULT_SAMPLE_RATE = 44100


def _get_default_device():
    """Gets the default torch device."""
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def _sample_next_token(logits_BCxV, temperature, top_p, cfg_filter_top_k=None) -> torch.Tensor:
    """Samples the next token based on logits, temperature, and top_p."""
    exec_device = logits_BCxV.device
    if temperature == 0.0: return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature
    if cfg_filter_top_k is not None:
        cfg_filter_top_k = min(cfg_filter_top_k, logits_BCxV.shape[-1])
        if cfg_filter_top_k > 0:
            _, top_k_indices_BCxV = torch.topk(logits_BCxV, k=cfg_filter_top_k, dim=-1)
            mask = torch.ones_like(logits_BCxV, dtype=torch.bool, device=exec_device)
            mask.scatter_(dim=-1, index=top_k_indices_BCxV, value=False)
            logits_BCxV = logits_BCxV.masked_fill(mask, -torch.inf)

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(probs_BCxV, dim=-1, descending=True)
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)
        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        sorted_indices_to_remove_BCxV[..., 1:] = sorted_indices_to_remove_BCxV[..., :-1].clone()
        sorted_indices_to_remove_BCxV[..., 0] = 0
        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV, device=exec_device)
        indices_to_remove_BCxV.scatter_(dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV)
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
    final_probs_BCxV = torch.nan_to_num(final_probs_BCxV)
    if torch.sum(final_probs_BCxV) == 0:
        final_probs_BCxV.fill_(1.0)
        final_probs_BCxV = final_probs_BCxV / final_probs_BCxV.shape[-1]

    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    return sampled_indices_BC.squeeze(-1)


class ComputeDtype(str, Enum):
    """Enum for compute dtypes."""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    def to_dtype(self) -> torch.dtype:
        """Converts enum value to torch.dtype."""
        return getattr(torch, self.value)


class Dia:
    """Main class for Dia TTS model loading and generation."""
    def __init__(self, config: DiaConfig, compute_dtype=ComputeDtype.FLOAT32, device=None):
        """Initializes the Dia model, ensuring placement on the specified device."""
        super().__init__()
        self.config = config
        self.device = device if device is not None else _get_default_device()
        if isinstance(compute_dtype, str): compute_dtype = ComputeDtype(compute_dtype)
        self.compute_dtype = compute_dtype.to_dtype()
        self.model = DiaModel(config, self.compute_dtype)
        self.model.to(self.device)
        self.dac_model = None

    def _devices_equal(self, device1: torch.device, device2: torch.device) -> bool:
        """Robustly compares two torch.device objects."""
        if device1 is None or device2 is None: return False
        if device1.type != device2.type: return False
        if device1.type == 'cuda':
            index1 = device1.index if device1.index is not None else torch.cuda.current_device()
            index2 = device2.index if device2.index is not None else torch.cuda.current_device()
            return index1 == index2
        return True

    def _load_dac_model(self):
        """Loads the Descript Audio Codec model to the instance's device."""
        if self.dac_model is not None:
             try:
                 dac_device = next(self.dac_model.parameters()).device
                 if not self._devices_equal(dac_device, self.device):
                     print(f"Moving existing DAC model from {dac_device} to {self.device}...")
                     self.dac_model.to(self.device)
             except StopIteration:
                 print("DAC model has no parameters, forcing reload.")
                 self.dac_model = None
             except Exception as e:
                  print(f"Error checking/moving DAC model device: {e}. Reloading.")
                  self.dac_model = None
             if self.dac_model is not None: return

        try:
            print(f"Loading DAC model to {self.device}...")
            dac_model_path = dac.utils.download()
            self.dac_model = dac.DAC.load(dac_model_path)
            self.dac_model.to(self.device)
            self.dac_model.eval()
            dac_device_check = next(self.dac_model.parameters()).device
            if not self._devices_equal(dac_device_check, self.device):
                 print(f"Warning: DAC model loaded to {dac_device_check} instead of requested {self.device}. Retrying move.")
                 self.dac_model.to(self.device)
                 dac_device_check_retry = next(self.dac_model.parameters()).device
                 if not self._devices_equal(dac_device_check_retry, self.device):
                      raise RuntimeError(f"Failed to move DAC model to {self.device}")
            print(f"DAC model loaded successfully on {self.device}.")
        except Exception as e:
            self.dac_model = None
            raise RuntimeError("Failed to load DAC model") from e

    def _prepare_text_input(self, text: str) -> torch.Tensor:
        """Encodes text prompt, pads, and creates tensor on the model's device."""
        text_pad_value = self.config.data.text_pad_value
        max_len = self.config.data.text_length
        try:
            byte_text = text.encode("utf-8")
            replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
            text_tokens = list(replaced_bytes)
        except Exception as e:
            print(f"Error encoding text: {e}")
            raise ValueError("Failed to encode input text.") from e

        current_len = len(text_tokens)
        padding_needed = max_len - current_len
        if padding_needed < 0:
            print(f"Warning: Input text truncated from {current_len} to {max_len} bytes.")
            text_tokens = text_tokens[:max_len]
            padded_text_np = np.array(text_tokens, dtype=np.uint8)
        elif padding_needed == 0:
             padded_text_np = np.array(text_tokens, dtype=np.uint8)
        else:
            padded_text_np = np.pad(text_tokens, (0, padding_needed), 'constant', constant_values=text_pad_value).astype(np.uint8)
        return torch.from_numpy(padded_text_np).to(dtype=torch.long, device=self.device).unsqueeze(0)

    def encode_audio_prompt(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Encodes a raw waveform tensor into DAC codes for use as an audio prompt."""
        if self.dac_model is None: self._load_dac_model()

        if waveform.device != self.device: waveform = waveform.to(self.device)
        if waveform.dtype != torch.float32:
            original_dtype = waveform.dtype
            waveform = waveform.to(torch.float32)
            if not torch.is_floating_point(original_dtype):
                 max_val = torch.iinfo(original_dtype).max
                 waveform = waveform / max_val

        if waveform.ndim == 2 and waveform.shape[0] == 1: waveform = waveform.unsqueeze(1)
        elif waveform.ndim == 1: waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 3 and waveform.shape[0] == 1 and waveform.shape[1] > 1:
             print(f"Dia.encode_audio_prompt: Input prompt has {waveform.shape[1]} channels. Averaging to mono.")
             waveform = torch.mean(waveform, dim=1, keepdim=True)
        elif not (waveform.ndim == 3 and waveform.shape[0] == 1 and waveform.shape[1] == 1):
            raise ValueError(f"Unsupported waveform shape for audio prompt: {waveform.shape}.")

        if sample_rate != DEFAULT_SAMPLE_RATE:
            print(f"Dia.encode_audio_prompt: Resampling prompt from {sample_rate} Hz to {DEFAULT_SAMPLE_RATE} Hz.")
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=DEFAULT_SAMPLE_RATE)
        waveform = torch.clamp(waveform, -1.0, 1.0)

        try:
            audio_data = self.dac_model.preprocess(waveform, DEFAULT_SAMPLE_RATE)
            _, codes, _, _, _ = self.dac_model.encode(audio_data)
            prompt_codes = codes.squeeze(0).transpose(0, 1).contiguous()
            return prompt_codes.to(torch.int)
        except Exception as e:
            print(f"Error during DAC encoding of prompt: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Failed to encode audio prompt using DAC.") from e

    def _prepare_audio_prompt(self, audio_prompt_codes: torch.Tensor | None) -> tuple[torch.Tensor, int]:
        """Prepares the initial audio tokens (BOS and optional prompt codes) on the model's device."""
        num_channels = self.config.data.channels
        audio_bos_value = self.config.data.audio_bos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        max_delay_pattern = max(delay_pattern)

        prefill = torch.full((1, num_channels), fill_value=audio_bos_value, dtype=torch.int, device=self.device)
        prefill_step = 1

        if audio_prompt_codes is not None:
            if audio_prompt_codes.device != self.device: audio_prompt_codes = audio_prompt_codes.to(self.device)
            if not torch.is_tensor(audio_prompt_codes) or audio_prompt_codes.dtype not in [torch.int, torch.long]:
                 audio_prompt_codes = audio_prompt_codes.to(torch.int)
            if audio_prompt_codes.ndim != 2 or audio_prompt_codes.shape[1] != num_channels:
                 raise ValueError(f"Audio prompt codes have unexpected shape {audio_prompt_codes.shape}.")
            prompt_len = audio_prompt_codes.shape[0]
            prefill_step += prompt_len
            prefill = torch.cat([prefill, audio_prompt_codes], dim=0)
            print(f"Prepared audio prompt with {prompt_len} timesteps.")

        delay_pad_tensor = torch.full((max_delay_pattern, num_channels), fill_value=-1, dtype=torch.int, device=self.device)
        prefill = torch.cat([prefill, delay_pad_tensor], dim=0)

        # Pass device to build_delay_indices
        delay_precomp = build_delay_indices(B=1, T=prefill.shape[0], C=num_channels, delay_pattern=delay_pattern, device=self.device)
        prefill_delayed = apply_audio_delay(prefill.unsqueeze(0), audio_pad_value, audio_bos_value, delay_precomp).squeeze(0)
        return prefill_delayed, prefill_step

    def _prepare_generation(self, text: str, audio_prompt_codes: torch.Tensor | None, verbose: bool):
        """Prepares encoder/decoder states and initial inputs, ensuring device consistency."""
        try:
            param_device = next(self.model.parameters()).device
            if not self._devices_equal(param_device, self.device):
                print(f"Warning: Model parameters on {param_device}, but expected {self.device}. Moving...")
                self.model.to(self.device)
                if not self._devices_equal(next(self.model.parameters()).device, self.device):
                     raise RuntimeError(f"Failed to move model parameters to {self.device}")
        except StopIteration: raise RuntimeError("Model has no parameters!")

        enc_input_cond = self._prepare_text_input(text)
        enc_input_uncond = torch.zeros_like(enc_input_cond)
        enc_input = torch.cat([enc_input_uncond, enc_input_cond], dim=0)
        prefill, prefill_step = self._prepare_audio_prompt(audio_prompt_codes)
        if verbose: print(f"generate: data loaded. Prefill steps: {prefill_step}. Prefill tensor shape: {prefill.shape}")

        enc_state = EncoderInferenceState.new(self.config, enc_input_cond)
        encoder_out = self.model.encoder(enc_input, enc_state)
        dec_cross_attn_cache = self.model.decoder.precompute_cross_attn_cache(encoder_out, enc_state.positions)
        dec_state = DecoderInferenceState.new(self.config, enc_state, encoder_out, dec_cross_attn_cache, self.compute_dtype)
        dec_output = DecoderOutput.new(self.config, self.device)
        dec_output.prefill(prefill, prefill_step)

        dec_step = prefill_step - 1
        if dec_step > 0:
            if verbose: print(f"generate: Prefilling decoder state for {dec_step} steps...")
            dec_state.prepare_step(0, dec_step)
            tokens_BxTxC = dec_output.get_tokens_at(0, dec_step).unsqueeze(0).expand(2, -1, -1)
            _ = self.model.decoder.forward(tokens_BxTxC, dec_state)
            if verbose: print("generate: Decoder prefill complete.")
        return dec_state, dec_output

    def _decoder_step(self, tokens_Bx1xC, dec_state, cfg_scale, temperature, top_p, cfg_filter_top_k) -> torch.Tensor:
        """Performs a single autoregressive decoding step."""
        audio_eos_value = self.config.data.audio_eos_value
        logits_Bx1xCxV = self.model.decoder.decode_step(tokens_Bx1xC, dec_state)
        logits_last_BxCxV = logits_Bx1xCxV[:, -1, :, :]
        uncond_logits_CxV = logits_last_BxCxV[0, :, :]
        cond_logits_CxV = logits_last_BxCxV[1, :, :]
        logits_CxV = cond_logits_CxV + cfg_scale * (cond_logits_CxV - uncond_logits_CxV)
        logits_CxV[:, audio_eos_value + 1 :] = -torch.inf
        logits_CxV[1:, audio_eos_value:] = -torch.inf
        return _sample_next_token(logits_CxV.float(), temperature, top_p, cfg_filter_top_k)

    def _generate_output(self, generated_codes: torch.Tensor) -> np.ndarray:
        """Reverts delay pattern and decodes audio codes using DAC."""
        if self.dac_model is None: self._load_dac_model()
        try:
            dac_device = next(self.dac_model.parameters()).device
            if not self._devices_equal(dac_device, self.device): self.dac_model.to(self.device)
        except StopIteration: raise RuntimeError("DAC model has no parameters after load attempt.")

        num_channels = self.config.data.channels
        seq_length = generated_codes.shape[0]
        delay_pattern = self.config.data.delay_pattern
        audio_pad_value = self.config.data.audio_pad_value
        max_delay_pattern = max(delay_pattern)

        # Pass the device of generated_codes to build_revert_indices
        codes_device = generated_codes.device
        revert_precomp = build_revert_indices(B=1, T=seq_length, C=num_channels, delay_pattern=delay_pattern, device=codes_device)

        codebook_delayed = generated_codes.unsqueeze(0)
        # Ensure generated codes are on the same device as precomputed indices
        # This shouldn't be necessary if build_revert_indices uses codes_device, but belts and braces
        if codebook_delayed.device != codes_device: codebook_delayed = codebook_delayed.to(codes_device)

        codebook_reverted = revert_audio_delay(codebook_delayed, audio_pad_value, revert_precomp, seq_length)
        codebook_trimmed = codebook_reverted[:, :-max_delay_pattern, :]

        min_valid_index = 0
        max_valid_index = 1023
        invalid_mask = (codebook_trimmed < min_valid_index) | (codebook_trimmed > max_valid_index)
        if torch.any(invalid_mask):
             print(f"Warning: Clamping {torch.sum(invalid_mask)} invalid codebook indices to 0.")
             codebook_trimmed[invalid_mask] = 0

        # codebook_for_dac should now be on the correct device (codes_device)
        codebook_for_dac = codebook_trimmed.transpose(1, 2)

        # decode function will handle moving codes to dac_model's device if necessary
        audio_tensor = decode(self.dac_model, codebook_for_dac)
        return audio_tensor.squeeze().cpu().numpy()


    @torch.inference_mode()
    def generate(self, text: str, max_tokens=None, cfg_scale=3.0, temperature=1.3, top_p=0.95,
                 use_torch_compile=False, cfg_filter_top_k=35, audio_prompt=None, verbose=False,
                 pbar=None
                 ) -> np.ndarray | None:
        """Generates audio waveform from text input, updating progress bars."""
        audio_eos_value = self.config.data.audio_eos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        clamped_max_tokens = self.config.data.audio_length if max_tokens is None else min(max_tokens, self.config.data.audio_length)
        max_delay_pattern = max(delay_pattern)
        self.model.eval()

        dec_state, dec_output = self._prepare_generation(text, audio_prompt, verbose)
        dec_step = dec_output.prefill_step - 1

        bos_countdown = max_delay_pattern
        eos_detected = False
        eos_countdown = -1
        step_fn = torch.compile(self._decoder_step, mode="default") if use_torch_compile else self._decoder_step

        tqdm_pbar = None
        if verbose:
            tqdm_total = clamped_max_tokens - (dec_step + 1) if clamped_max_tokens > dec_step + 1 else 0
            tqdm_pbar = tqdm(total=tqdm_total, desc="Dia Generating Tokens", unit="token", initial=0)

        while dec_step < clamped_max_tokens -1 :
            dec_state.prepare_step(dec_step)
            tokens_Bx1xC = dec_output.get_tokens_at(dec_step).unsqueeze(0).expand(2, -1, -1)
            pred_C = step_fn(tokens_Bx1xC, dec_state, cfg_scale, temperature, top_p, cfg_filter_top_k)

            is_last_generatable_step = (dec_step == clamped_max_tokens - max_delay_pattern - 1)
            if (not eos_detected and pred_C[0] == audio_eos_value) or is_last_generatable_step:
                if not eos_detected:
                     if verbose: print(f"\nEOS detected at step {dec_step + 1}.")
                     eos_detected = True
                     eos_countdown = max_delay_pattern
                if is_last_generatable_step and not eos_detected:
                     if verbose: print(f"\nApproaching max_tokens ({clamped_max_tokens}), forcing EOS generation.")
                     pred_C[0] = audio_eos_value
                     eos_detected = True
                     eos_countdown = max_delay_pattern

            if eos_countdown > 0:
                step_after_eos = max_delay_pattern - eos_countdown
                for i, d in enumerate(delay_pattern):
                    if step_after_eos == d: pred_C[i] = audio_eos_value
                    elif step_after_eos > d: pred_C[i] = audio_pad_value
                eos_countdown -= 1

            bos_countdown = max(0, bos_countdown - 1)
            dec_output.update_one(pred_C, dec_step + 1, apply_mask=(bos_countdown > 0))
            if eos_countdown == 0:
                if verbose: print("EOS padding complete. Stopping generation.")
                break

            dec_step += 1
            if pbar: pbar.update(1)
            if tqdm_pbar: tqdm_pbar.update(1)

        if tqdm_pbar: tqdm_pbar.close()
        final_step_index = dec_step
        if dec_output.prefill_step > final_step_index:
            print("Dia: Warning - Nothing generated beyond prefill/prompt.")
            return None

        generated_codes = dec_output.generated_tokens[dec_output.prefill_step : final_step_index + 1, :]
        if generated_codes.shape[0] == 0:
            print("Dia: Warning - Generated codes array is empty.")
            return None
        if verbose: print(f"Generated {generated_codes.shape[0]} token steps.")

        return self._generate_output(generated_codes)