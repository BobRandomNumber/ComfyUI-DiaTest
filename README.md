# ComfyUI DiaTest TTS Node

This node pack integrates the [Nari Labs Dia](https://github.com/nari-labs/dia) text-to-speech model into ComfyUI using separate nodes for loading the model from a local file and generating audio.

Dia allows generating dialogue with speaker tags (`[S1]`, `[S2]`) and non-verbal sounds (`(laughs)`, etc.). This node pack loads the model from a local `.safetensors` file onto the GPU and generates audio. It **requires a CUDA-enabled GPU**.

**Note:** This version is specifically configured for the `nari-labs/Dia-1.6B` model architecture.

## Installation

1.  Ensure you have a CUDA-enabled GPU and the necessary NVIDIA drivers installed.
2.  Download the Dia-1.6B model weights file (`dia-1.6B.safetensors`) from Hugging Face Hub:
    *   **Direct Download URL:** [https://huggingface.co/nari-labs/Dia-1.6B/resolve/main/dia-1.6B.safetensors](https://huggingface.co/nari-labs/Dia-1.6B/resolve/main/dia-1.6B.safetensors)
3.  Place the downloaded `.safetensors` file into one of the directories ComfyUI recognizes for `diffusion_models` (e.g., `ComfyUI/models/diffusion_models/`). You might want to rename it to `Dia-1.6B.safetensors` for clarity.
4.  Navigate to your `ComfyUI/custom_nodes/` directory.
5.  Clone this repository:
    ```bash
    git clone https://github.com/BobRandomNumber/ComfyUI-DiaTest.git
    ```
    Alternatively, download the ZIP and extract it into `custom_nodes`.
6.  Install the required dependencies:
    *   Activate ComfyUI's Python environment (e.g., `source ./venv/bin/activate`).
    *   Navigate to the node directory: `cd ComfyUI/custom_nodes/ComfyUI-DiaTest`
    *   Install requirements: `pip install -r requirements.txt`
7.  Restart ComfyUI.

## Nodes

### Dia 1.6b Loader (`DiaLoader`)

Loads the Dia-1.6B TTS model from a local `.safetensors` checkpoint file located in one of your configured `diffusion_models` directories. Loads the model weights and the required DAC codec onto the GPU using the embedded Dia-1.6B configuration.

**Inputs:**

*   `ckpt_name`: Dropdown list of found `.safetensors` files within your `diffusion_models` directories. Select the file corresponding to the Dia-1.6B model.

**Outputs:**

*   `dia_model`: A custom `DIA_MODEL` object containing the loaded Dia model instance, ready for the `DiaGenerate` node.

### Dia TTS Generate (`DiaGenerate`)

Generates audio using a pre-loaded Dia model provided by the `DiaLoader` node. Displays a progress bar during generation.

**Inputs:**

*   `dia_model`: The `DIA_MODEL` output from the `DiaLoader` node.
*   `text`: The main text transcript to generate audio for. Use `[S1]`, `[S2]` for speaker turns and parentheses for non-verbals like `(laughs)`.
*   `max_tokens`: Maximum number of audio tokens to generate (controls length).
*   `cfg_scale`: Classifier-Free Guidance scale.
*   `temperature`: Sampling temperature.
*   `top_p`: Nucleus sampling probability.
*   `cfg_filter_top_k`: Top-K filtering applied during CFG.
*   `speed_factor`: Adjusts the speed of the generated audio (1.0 = original speed).
*   `seed`: Random seed for reproducibility.

**Outputs:**

*   `audio`: The generated audio (`AUDIO` format: `{'waveform': tensor[B,C,T], 'sample_rate': sr}`), ready to be saved or previewed. Sample rate is 44100 Hz.

## Usage Example

1.  Download `dia-1.6B.safetensors` and place it in a `diffusion_models` directory (e.g., `ComfyUI/models/diffusion_models/`).
2.  Add the `Dia 1.6b Loader` node from the `audio/DiaTest` category.
3.  Select your Dia model file (e.g., `dia-1.6B.safetensors`) from the `ckpt_name` dropdown.
4.  Add the `Dia TTS Generate` node (also from `audio/DiaTest`).
5.  Connect the `dia_model` output of the Loader node to the `dia_model` input of the Generate node.
6.  Enter your dialogue script into the `text` input on the Generate node.
7.  Adjust generation parameters on the Generate node as needed.
8.  Connect the `audio` output of the Generate node to a `SaveAudio` or `PreviewAudio` node.
9.  Queue the prompt.

## Notes

*   This node pack **requires a CUDA-enabled GPU**.
*   Only the `.safetensors` weights file is required.
*   The first time you load a specific model with the `DiaLoader`, it loads the weights and the required DAC codec. Subsequent runs using the same loader node will use the cached model object. Switching models in the loader clears the cache.
*   Dependencies `descript-audio-codec`, `huggingface_hub`, and `safetensors` must be installed via `requirements.txt`.