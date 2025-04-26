# ComfyUI DiaTest TTS Node

Warning LLM Code there are probably better options

This node pack partially integrates the [Nari Labs Dia](https://github.com/nari-labs/dia) text-to-speech model into ComfyUI using a single node for loading (onto GPU) and generation.

Dia allows generating dialogue with speaker tags (`[S1]`, `[S2]`) and non-verbal sounds (`(laughs)`, etc.). This node loads the model from Hugging Face Hub and generates audio directly using float32 precision. It requires a CUDA-enabled GPU.

## Installation

1.  Ensure you have a CUDA-enabled GPU and the necessary NVIDIA drivers installed.
2.  Navigate to your `ComfyUI/custom_nodes/` directory.
3.  Clone this repository:
    ```bash
    git clone <repository_url> ComfyUI-DiaTest
    ```
    (Replace `<repository_url>` with the actual URL).
    Alternatively, download the ZIP and extract it into `custom_nodes` as `ComfyUI-DiaTest`.
4.  Install the required dependencies:
    *   Activate ComfyUI's Python environment (e.g., `source ./venv/bin/activate`).
    *   Navigate to the node directory: `cd ComfyUI/custom_nodes/ComfyUI-DiaTest`
    *   Install requirements: `pip install -r requirements.txt`
5.  Restart ComfyUI.

## Node

### Dia TTS Generate (`DiaGenerate`)

Loads the specified Dia model from Hugging Face Hub onto the GPU (if not already cached) and generates audio based on the provided text and parameters. Uses float32 precision internally. Requires CUDA.

**Inputs:**

*   `repo_id`: The Hugging Face repository ID (default: `nari-labs/Dia-1.6B`).
*   `text`: The main text transcript to generate audio for. Use `[S1]`, `[S2]` for speaker turns and parentheses for non-verbals like `(laughs)`.
*   `max_tokens`: Maximum number of audio tokens to generate (controls length).
*   `cfg_scale`: Classifier-Free Guidance scale (higher means stronger adherence to text).
*   `temperature`: Sampling temperature (higher means more randomness).
*   `top_p`: Nucleus sampling probability.
*   `cfg_filter_top_k`: Top-K filtering applied during CFG.
*   `speed_factor`: Adjusts the speed of the generated audio (1.0 = original speed, <1.0 = slower, >1.0 = faster). Default: 0.94.
*   `seed`: Random seed for reproducibility.

**Outputs:**

*   `audio`: The generated audio (`AUDIO` format: `{'waveform': tensor[B,C,T], 'sample_rate': sr}`), ready to be saved or previewed. Sample rate is 44100 Hz.

## Usage Example

1.  Add the `Dia TTS Generate` node from the `audio/DiaTest` category.
2.  Configure the `repo_id` if needed.
3.  Enter your dialogue script into the `text` input.
4.  Adjust generation parameters (`cfg_scale`, `temperature`, `speed_factor`, etc.) as needed.
5.  Connect the `audio` output to a `SaveAudio` or `PreviewAudio` node.
6.  Queue the prompt.

## Notes

*   This node **requires a CUDA-enabled GPU**. It will fail to load if CUDA is not detected.
*   The first time you run the node for a specific `repo_id`, it will download the model files from Hugging Face Hub, which may take some time. Subsequent runs will use the cached model.
*   Changing the `repo_id` will trigger a model reload.
*   The model uses float32 precision internally.
*   The Descript Audio Codec (DAC) dependency (`descript-audio-codec`) must be installed via `requirements.txt`.
