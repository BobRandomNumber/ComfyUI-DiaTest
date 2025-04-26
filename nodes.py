# ComfyUI-DiaTest/nodes.py

import os
import torch
import numpy as np
import folder_paths
import torchaudio # Kept for potential internal use by dia_lib
from huggingface_hub import hf_hub_download
import traceback
import gc

# --- Import Dia library components ---
try:
    from .dia_lib.model import Dia, ComputeDtype, DEFAULT_SAMPLE_RATE
    from .dia_lib.config import DiaConfig
except ImportError as e:
    print("ComfyUI-DiaTest: Error importing Dia library components.")
    print(f"Ensure the 'dia_lib' folder exists in '{os.path.dirname(__file__)}'.")
    print(f"Import error: {e}")
    raise e
# --- End Dia library imports ---

# --- Helper Functions ---
def get_torch_device():
    """Checks for CUDA availability and returns the CUDA device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        # If CUDA is not available, raise an error as GPU is required.
        raise RuntimeError("CUDA device not available. This node requires a CUDA-enabled GPU.")
# --- End Helper Functions ---

# --- Global model cache ---
loaded_dia_model = None
loaded_model_key = None

class DiaGenerate:
    """Loads the Dia TTS model from Hub onto GPU and generates audio."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Model Loading Params
                "repo_id": ("STRING", {"default": "nari-labs/Dia-1.6B"}),
                # No device override - GPU is forced

                # Generation Params
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "[S1] Hello world. [S2] This is a test."}),
                "max_tokens": ("INT", {"default": 1720, "min": 860, "max": 3072, "step": 10}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 7.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.3, "min": 0.1, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "cfg_filter_top_k": ("INT", {"default": 35, "min": 1, "max": 100, "step": 1}),
                "speed_factor": ("FLOAT", {"default": 0.94, "min": 0.5, "max": 1.5, "step": 0.01}), # Added speed factor
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_and_generate"
    CATEGORY = "audio/DiaTest"

    def load_and_generate(self, repo_id, text: str, max_tokens: int, cfg_scale: float, temperature: float, top_p: float, cfg_filter_top_k: int, speed_factor: float, seed: int):
        global loaded_dia_model, loaded_model_key

        # --- Model Loading Logic ---
        # Force GPU device
        device = get_torch_device() # This will raise error if no CUDA

        # Use float32 internally
        compute_dtype_str = "float32"

        print(f"DiaTestGenerate: Target device: {device}, Compute dtype: {compute_dtype_str}")

        # Cache key includes repo and dtype (device is fixed to CUDA)
        current_key = (repo_id, compute_dtype_str, str(device))

        dia_model = None
        # Check cache
        if loaded_dia_model is not None and current_key == loaded_model_key:
            print("DiaTestGenerate: Using cached model.")
            dia_model = loaded_dia_model
            # Defensive check: ensure cached model is indeed on CUDA
            if dia_model.device != device:
                 print(f"DiaTestGenerate: Warning: Cached model not on expected device ({dia_model.device}). Moving to {device}.")
                 try:
                     dia_model.model.to(device)
                     if dia_model.dac_model: dia_model.dac_model.to(device)
                     dia_model.device = device
                 except Exception as move_e:
                     print(f"DiaTestGenerate: Error moving cached model: {move_e}")
                     loaded_dia_model = None; loaded_model_key = None; raise move_e
        else:
            # Clear previous model if config changed
            if loaded_dia_model is not None:
                print(f"DiaTestGenerate: Configuration changed. Clearing previous model...")
                try:
                    if hasattr(loaded_dia_model, 'model'): del loaded_dia_model.model
                    if hasattr(loaded_dia_model, 'dac_model'): del loaded_dia_model.dac_model
                    del loaded_dia_model
                except Exception as del_e: print(f"DiaTestGenerate: Error deleting previous model: {del_e}")
                loaded_dia_model = None; loaded_model_key = None; gc.collect()
                torch.cuda.empty_cache(); print("DiaTestGenerate: Cleared CUDA cache.")

            # Load model from Hub
            print(f"DiaTestGenerate: Loading model from Hugging Face Hub: repo_id='{repo_id}'")
            try:
                # Pre-download files (optional)
                try:
                    cache_dir = os.path.join(folder_paths.models_dir, "huggingface")
                    os.makedirs(cache_dir, exist_ok=True)
                    hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=cache_dir, resume_download=True, etag_timeout=10)
                    hf_hub_download(repo_id=repo_id, filename="dia-v0_1.pth", cache_dir=cache_dir, resume_download=True, etag_timeout=10)
                    print(f"DiaTestGenerate: Ensured model files are cached.")
                except Exception as download_e:
                    print(f"DiaTestGenerate: Warning during file pre-check/download: {download_e}")

                # Load the model
                dia_model = Dia.from_pretrained(
                    model_name=repo_id,
                    compute_dtype=compute_dtype_str, # Use string 'float32'
                    device=device # Load directly onto GPU
                )
                print("DiaTestGenerate: Model loaded successfully.")
                loaded_dia_model = dia_model
                loaded_model_key = current_key

            except Exception as e:
                print(f"DiaTestGenerate: Error loading model: {e}")
                traceback.print_exc()
                loaded_dia_model = None; loaded_model_key = None
                raise e
        # --- End Model Loading Logic ---


        # --- Generation Logic ---
        if not text or text.isspace(): raise ValueError("Input text cannot be empty.")

        # Seed setting
        MAX_SEED_NUMPY = 2**32 - 1
        seed_torch = seed; seed_numpy = seed % MAX_SEED_NUMPY
        torch.manual_seed(seed_torch); np.random.seed(seed_numpy)
        torch.cuda.manual_seed_all(seed_torch) # Seed CUDA
        print(f"DiaTestGenerate: Using ComfyUI seed {seed} (Torch: {seed_torch}, NumPy: {seed_numpy})")

        text_for_generate = text

        try:
            print(f"DiaTestGenerate: Starting generation...")

            # Generation Call
            with torch.inference_mode():
                output_np = dia_model.generate(
                    text=text_for_generate, max_tokens=max_tokens, cfg_scale=cfg_scale, temperature=temperature,
                    top_p=top_p, cfg_filter_top_k=cfg_filter_top_k,
                    # No audio_prompt
                    use_torch_compile=False, verbose=True
                )

            # Handle failed generation
            if output_np is None or output_np.size == 0:
                print("DiaTestGenerate: Warning - Generation returned None or empty array. Outputting silence.")
                silent_tensor = torch.zeros((1, 1, DEFAULT_SAMPLE_RATE), dtype=torch.float32)
                result = {'waveform': silent_tensor, 'sample_rate': DEFAULT_SAMPLE_RATE}
                return (result,)

            print(f"DiaTestGenerate: Raw generation complete. Shape: {output_np.shape}")

            # --- Apply Speed Factor ---
            if speed_factor != 1.0:
                # Ensure speed_factor is valid
                speed_factor = max(0.1, min(speed_factor, 5.0)) # Clamp to reasonable range
                original_len = len(output_np)
                target_len = int(original_len / speed_factor)

                if target_len > 0 and target_len != original_len:
                    print(f"DiaTestGenerate: Applying speed factor {speed_factor:.2f}x (length {original_len} -> {target_len})")
                    x_original = np.arange(original_len)
                    x_resampled = np.linspace(0, original_len - 1, target_len)
                    # Ensure float input for interp if not already
                    if not np.issubdtype(output_np.dtype, np.floating):
                         output_np = output_np.astype(np.float32)
                    resampled_audio_np = np.interp(x_resampled, x_original, output_np)
                    output_np = resampled_audio_np # Use the resampled audio
                else:
                    print(f"DiaTestGenerate: Skipping speed adjustment (factor: {speed_factor:.2f}).")
            # --- End Speed Factor ---


            # --- Output Formatting ---
            try:
                # Convert numpy to tensor, ensure float32
                output_tensor = torch.from_numpy(output_np.astype(np.float32))

                # Ensure shape [channels, samples]
                if output_tensor.ndim == 1: # Mono [samples] -> [1, samples]
                    output_tensor = output_tensor.unsqueeze(0)
                elif output_tensor.ndim != 2: # Should only be 1D or 2D at this point
                    raise ValueError(f"Unexpected audio array dimension after speed factor: {output_tensor.ndim}.")
                # Assuming 2D is already [channels, samples] - interpolation keeps channel dim first if input was 2D

                # Add batch dimension -> [1, channels, samples]
                output_tensor = output_tensor.unsqueeze(0)
                output_tensor = output_tensor.contiguous()

                # Final log & sanity checks
                final_shape = output_tensor.shape; final_dtype = output_tensor.dtype
                print(f"DiaTestGenerate: Final audio tensor shape: {final_shape}, dtype: {final_dtype}")
                if len(final_shape) != 3: raise ValueError(f"Internal Error: Final tensor dim not 3! Shape: {final_shape}")
                if final_shape[0] != 1: print(f"DiaTestGenerate: Warning - final batch size not 1: {final_shape[0]}")
                if final_shape[1] == 0 or final_shape[2] == 0: raise ValueError(f"Internal Error: Final tensor zero dim! Shape: {final_shape}")

                # Create dictionary for ComfyUI AUDIO output type
                result = {'waveform': output_tensor, 'sample_rate': DEFAULT_SAMPLE_RATE}
                return (result,) # Return dict inside tuple

            except Exception as format_e:
                print(f"DiaTestGenerate: Error formatting output: {format_e}")
                traceback.print_exc()
                raise format_e
            # --- End Output Formatting ---

        except Exception as e:
            print(f"DiaTestGenerate: Error during generation: {e}")
            traceback.print_exc()
            raise e


# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "DiaGenerate": DiaGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiaGenerate": "Dia TTS Generate",
}

# --- Print message on load ---
print("### Loading: ComfyUI-DiaTest Nodes ###")