# ComfyUI-DiaTTS/nodes.py

import os
import torch
import numpy as np
import folder_paths
import traceback
import gc
from safetensors.torch import load_file as load_safetensors_file
import comfy.utils
import torchaudio # Needed for potential resampling in encode_audio_prompt

# --- Import Dia library components ---
try:
    from .dia_lib.model import Dia, ComputeDtype, DEFAULT_SAMPLE_RATE
    from .dia_lib.config import DiaConfig
    from .dia_lib.state import EncoderInferenceState, DecoderInferenceState, DecoderOutput
except ImportError as e:
    print("ComfyUI-DiaTest: Error importing Dia library components.")
    print(f"Ensure the 'dia_lib' folder exists in '{os.path.dirname(__file__)}'.")
    print(f"Import error: {e}")
    raise e

# --- Hardcoded Config for Dia-1.6B ---
DEFAULT_DIA_1_6B_CONFIG = {
  "data": { "audio_bos_value": 1026, "audio_eos_value": 1024, "audio_length": 3072, "audio_pad_value": 1025, "channels": 9, "delay_pattern": [0, 8, 9, 10, 11, 12, 13, 14, 15], "text_length": 1024, "text_pad_value": 0 },
  "model": { "decoder": { "cross_head_dim": 128, "cross_query_heads": 16, "gqa_head_dim": 128, "gqa_query_heads": 16, "kv_heads": 4, "n_embd": 2048, "n_hidden": 8192, "n_layer": 18 }, "dropout": 0.0, "encoder": { "head_dim": 128, "n_embd": 1024, "n_head": 16, "n_hidden": 4096, "n_layer": 12 }, "normalization_layer_epsilon": 1e-05, "rope_max_timescale": 10000, "rope_min_timescale": 1, "src_vocab_size": 256, "tgt_vocab_size": 1028, "weight_dtype": "float32" },
  "training": {},
  "version": "0.1"
}


# --- Helper Functions ---
def get_torch_device():
    """Checks for CUDA availability and returns the CUDA device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        raise RuntimeError("CUDA device not available. Dia nodes require a CUDA-enabled GPU.")

# --- Global model cache ---
loaded_dia_objects = {}

class DiaLoader:
    """Loads the Dia-1.6B TTS model from a local safetensors file."""
    @classmethod
    def INPUT_TYPES(s):
        """Finds .safetensors files in diffusion_models directories."""
        try:
            safetensors_files = folder_paths.get_filename_list("diffusion_models")
            s.dia_model_files = sorted([f for f in safetensors_files if f.lower().endswith(".safetensors")])
            if not s.dia_model_files:
                print("DiaLoader: No .safetensors files found in diffusion_models directories.")
                s.dia_model_files = ["None"]
        except Exception as e:
            print(f"DiaLoader: Warning - Could not access diffusion_models paths: {e}")
            s.dia_model_files = ["None"]

        return { "required": { "ckpt_name": (s.dia_model_files,), }, }

    RETURN_TYPES = ("DIA_MODEL",)
    RETURN_NAMES = ("dia_model",)
    FUNCTION = "load_dia_model"
    CATEGORY = "audio/DiaTTS"

    def load_dia_model(self, ckpt_name: str):
        """Loads the safetensors weights, combines with embedded config, loads DAC, and prepares the Dia object."""
        global loaded_dia_objects

        if ckpt_name == "None": raise ValueError("No Dia model selected in DiaLoader.")

        ckpt_path = folder_paths.get_full_path("diffusion_models", ckpt_name)
        if not ckpt_path or not os.path.exists(ckpt_path):
             found = False
             for directory in folder_paths.get_folder_paths("diffusion_models"):
                 potential_path = os.path.join(directory, ckpt_name)
                 if os.path.exists(potential_path):
                     ckpt_path = potential_path; found = True; break
             if not found: raise FileNotFoundError(f"Checkpoint file '{ckpt_name}' not found.")

        device = get_torch_device()
        compute_dtype = torch.float32 # Dia currently configured for float32 compute

        current_key = (ckpt_path, str(compute_dtype), str(device))

        dia_object = None
        if current_key in loaded_dia_objects:
            print(f"DiaLoader: Using cached Dia object for '{ckpt_name}'.")
            dia_object = loaded_dia_objects[current_key]
            # Re-check device and DAC just in case
            if not dia_object.model._devices_equal(dia_object.device, device):
                 print(f"DiaLoader: Moving cached model from {dia_object.device} to {device}.")
                 try:
                     dia_object.model.to(device)
                     if dia_object.dac_model: dia_object.dac_model.to(device)
                     dia_object.device = device
                 except Exception as move_e:
                     print(f"DiaLoader: Error moving cached model: {move_e}")
                     if current_key in loaded_dia_objects: del loaded_dia_objects[current_key]
                     raise move_e
            if dia_object.dac_model is None:
                 print("DiaLoader: Cached object missing DAC model, attempting reload...")
                 try: dia_object._load_dac_model()
                 except Exception as dac_e: print(f"DiaLoader: Error loading DAC model for cached object: {dac_e}"); raise dac_e
            elif not dia_object.model._devices_equal(dia_object.dac_model.device, device):
                 print(f"DiaLoader: Moving cached DAC from {dia_object.dac_model.device} to {device}.")
                 try: dia_object.dac_model.to(device)
                 except Exception as dac_move_e: print(f"DiaLoader: Error moving cached DAC model: {dac_move_e}")
        else:
            if loaded_dia_objects:
                 print(f"DiaLoader: Different model requested ('{ckpt_name}'). Clearing cache.")
                 loaded_dia_objects.clear(); gc.collect(); torch.cuda.empty_cache()

            print(f"DiaLoader: Loading Dia-1.6B model configuration...")
            try: config = DiaConfig.model_validate(DEFAULT_DIA_1_6B_CONFIG)
            except Exception as config_e: print(f"DiaLoader: Error validating embedded config: {config_e}"); raise config_e

            print(f"DiaLoader: Instantiating Dia model on device={device}...")
            # Pass compute dtype string directly
            dia_object = Dia(config, compute_dtype=str(compute_dtype).split('.')[-1], device=device)

            print(f"DiaLoader: Loading model weights from: {ckpt_path}")
            try:
                # Load directly to target device to save memory
                state_dict = load_safetensors_file(ckpt_path, device=str(device))
                missing_keys, unexpected_keys = dia_object.model.load_state_dict(state_dict, strict=True)
                if missing_keys: print(f"DiaLoader: Warning - Missing keys in state_dict: {missing_keys}")
                if unexpected_keys: print(f"DiaLoader: Warning - Unexpected keys in state_dict: {unexpected_keys}")
                print("DiaLoader: Model weights loaded successfully.")
                del state_dict; gc.collect() # Clean up state dict
            except Exception as e:
                print(f"DiaLoader: Error loading state_dict: {e}"); traceback.print_exc()
                raise e

            # Load DAC model after main model to ensure correct device placement context
            try: dia_object._load_dac_model()
            except Exception as dac_e: print(f"DiaLoader: Error loading required DAC model: {dac_e}"); traceback.print_exc(); raise dac_e

            dia_object.model.eval()
            if dia_object.dac_model: dia_object.dac_model.eval()

            print(f"DiaLoader: Caching loaded Dia object.")
            loaded_dia_objects[current_key] = dia_object

        return (dia_object,)


class DiaGenerate:
    """Generates audio using a pre-loaded Dia TTS model, optionally with an audio prompt."""
    @classmethod
    def INPUT_TYPES(s):
        """Defines the inputs for audio generation."""
        return {
            "required": {
                "dia_model": ("DIA_MODEL",),
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "[S1] Hello world. [S2] This is a test."
                }),
                "max_tokens": ("INT", {"default": 1720, "min": 860, "max": 3072, "step": 10}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 7.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.3, "min": 0.1, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "cfg_filter_top_k": ("INT", {"default": 35, "min": 1, "max": 100, "step": 1}),
                "speed_factor": ("FLOAT", {"default": 0.94, "min": 0.5, "max": 1.5, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
             "optional": {
                "audio_prompt": ("AUDIO",), # Optional audio prompt input
             }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_audio"
    CATEGORY = "audio/DiaTTS"

    def generate_audio(self, dia_model: Dia, text: str, max_tokens: int, cfg_scale: float, temperature: float, top_p: float, cfg_filter_top_k: int, speed_factor: float, seed: int, audio_prompt=None):
        """
        Performs TTS generation. If audio_prompt is provided, the 'text' input
        should contain the transcript of the audio_prompt followed by the text to generate.
        """
        if dia_model is None: raise ValueError("Dia model object is required.")
        if not isinstance(dia_model, Dia): raise TypeError("Invalid object passed as dia_model.")
        if dia_model.model is None: raise ValueError("Dia object missing model.")
        # DAC model loading is handled internally by encode_audio_prompt or generate if needed

        exec_device = dia_model.device
        if exec_device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("Model is on CUDA but CUDA is not available.")

        clamped_max_tokens = min(max_tokens, dia_model.config.data.audio_length)
        if max_tokens > clamped_max_tokens:
             print(f"DiaGenerate: Clamping max_tokens ({max_tokens}) to model max ({clamped_max_tokens}).")
        max_tokens = clamped_max_tokens # Use the clamped value

        if not text or text.isspace(): raise ValueError("Input text cannot be empty.")

        # --- Handle Optional Audio Prompt ---
        encoded_audio_prompt = None
        if audio_prompt is not None:
            waveform = audio_prompt.get('waveform')
            sample_rate = audio_prompt.get('sample_rate')
            if waveform is not None and sample_rate is not None:
                # Ensure DAC is loaded before encoding
                if not dia_model.dac_model:
                    try: dia_model._load_dac_model()
                    except Exception as dac_e:
                        print(f"DiaGenerate: Error loading DAC model for prompt encoding: {dac_e}")
                        raise dac_e

                print("DiaGenerate: Encoding provided audio prompt...")
                try:
                    # Make sure waveform is float32 for DAC/resampling
                    if waveform.dtype != torch.float32:
                        waveform = waveform.to(torch.float32)
                        # Normalize if it was int type
                        original_dtype = audio_prompt.get('waveform').dtype
                        if not torch.is_floating_point(original_dtype):
                            max_val = torch.iinfo(original_dtype).max
                            waveform = waveform / max_val

                    encoded_audio_prompt = dia_model.encode_audio_prompt(waveform, sample_rate)
                    print(f"DiaGenerate: Audio prompt encoded successfully, shape: {encoded_audio_prompt.shape}")
                    print("DiaGenerate: Using the 'text' input directly as combined prompt transcript + generation text.")

                except Exception as encode_e:
                    print(f"DiaGenerate: Warning - Failed to encode audio prompt: {encode_e}")
                    traceback.print_exc()
                    encoded_audio_prompt = None # Proceed without prompt if encoding fails
            else:
                 print("DiaGenerate: Warning - Invalid audio_prompt dictionary received (missing waveform or sample_rate). Ignoring prompt.")

        # The 'text' input is used directly whether or not a prompt is provided
        text_for_generate = text

        # --- Set Seed ---
        MAX_SEED_NUMPY = 2**32 - 1
        seed_torch = seed; seed_numpy = seed % MAX_SEED_NUMPY
        torch.manual_seed(seed_torch); np.random.seed(seed_numpy)
        if exec_device.type == 'cuda': torch.cuda.manual_seed_all(seed_torch)

        # --- Progress Bar Setup ---
        pbar = comfy.utils.ProgressBar(max_tokens)

        try:
            print(f"DiaGenerate: Starting generation on {exec_device}...")
            # Pass the encoded prompt tensor (or None) to the generate method
            output_np = dia_model.generate(
                text=text_for_generate,
                max_tokens=max_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                use_torch_compile=False, # Keep False for ComfyUI stability
                verbose=True, # Enable Dia's internal verbose logging
                audio_prompt=encoded_audio_prompt, # Pass the encoded tensor or None
                pbar=pbar # Pass ComfyUI pbar object
            )

            if output_np is None or output_np.size == 0:
                print("DiaGenerate: Warning - Generation returned empty. Outputting silence.")
                silent_tensor = torch.zeros((1, 1, DEFAULT_SAMPLE_RATE), dtype=torch.float32) # Shape [1, 1, T]
                return ({'waveform': silent_tensor, 'sample_rate': DEFAULT_SAMPLE_RATE},)

            # --- Speed Factor Adjustment ---
            if speed_factor != 1.0:
                speed_factor = max(0.1, min(speed_factor, 5.0)) # Clamp speed factor
                original_len = len(output_np)
                target_len = int(original_len / speed_factor)
                if target_len > 0 and target_len != original_len:
                    print(f"DiaGenerate: Applying speed factor {speed_factor:.2f}x")
                    # Ensure float dtype for interpolation
                    if not np.issubdtype(output_np.dtype, np.floating):
                        output_np = output_np.astype(np.float32)

                    x_original = np.arange(original_len)
                    x_resampled = np.linspace(0, original_len - 1, target_len)
                    resampled_audio_np = np.interp(x_resampled, x_original, output_np)
                    output_np = resampled_audio_np # Use the resampled audio
                elif target_len == 0:
                     print(f"DiaGenerate: Warning - Speed factor {speed_factor:.2f}x results in zero length audio. Skipping adjustment.")
                else:
                     # No change in length or factor is 1.0
                     pass

            # --- Format Output for ComfyUI ---
            try:
                # Convert final numpy array to tensor
                output_tensor = torch.from_numpy(output_np.astype(np.float32))
                # Ensure correct shape [Batch, Channels, Samples] - Dia output is mono.
                if output_tensor.ndim == 1: # [T] -> [1, 1, T]
                    output_tensor = output_tensor.unsqueeze(0).unsqueeze(0)
                elif output_tensor.ndim == 2 and output_tensor.shape[0] == 1: # [1, T] -> [1, 1, T]
                    output_tensor = output_tensor.unsqueeze(1)
                elif output_tensor.ndim != 3 or output_tensor.shape[0] != 1 or output_tensor.shape[1] != 1:
                     raise ValueError(f"Unexpected intermediate audio tensor shape: {output_tensor.shape}. Expected mono audio resulting in [1, 1, T].")

                output_tensor = output_tensor.contiguous()

                print(f"DiaGenerate: Final audio tensor shape: {output_tensor.shape}, Sample Rate: {DEFAULT_SAMPLE_RATE}")
                if len(output_tensor.shape) != 3: raise ValueError("Final tensor dim not 3!")
                if output_tensor.shape[0] == 0 or output_tensor.shape[1] == 0 or output_tensor.shape[2] == 0:
                    raise ValueError(f"Final tensor has a zero dimension: {output_tensor.shape}")

                result = {'waveform': output_tensor, 'sample_rate': DEFAULT_SAMPLE_RATE}
                return (result,)
            except Exception as format_e:
                print(f"DiaGenerate: Error formatting output: {format_e}"); traceback.print_exc(); raise format_e

        except Exception as e:
            print(f"DiaGenerate: Error during generation: {e}")
            traceback.print_exc()
            raise e
        finally:
            # Clean up CUDA cache if needed after generation
            if exec_device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "DiaLoader": DiaLoader,
    "DiaGenerate": DiaGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiaLoader": "Dia 1.6b Loader",
    "DiaGenerate": "Dia TTS Generate",
}