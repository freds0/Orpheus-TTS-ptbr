# Orpheus-TTS-ptbr

This repository contains scripts for fine-tuning and running inference with an [Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS) model. It utilizes the SNAC codec for audio tokenization/detokenization and leverages Unsloth for efficient LoRA fine-tuning.


All code is based on Unsloath's Jupyter Notebook, available [here](https://github.com/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb).

## Features

*   **Text-to-Speech:** Generates speech audio from input text.
*   **Orpheus Architecture:** Based on the Orpheus TTS model.
*   **SNAC Codec:** Uses the SNAC neural audio codec (24kHz) for representing audio.
*   **Unsloth Optimization:** Employs Unsloth's `FastLanguageModel` for significantly faster training and potentially lower memory usage via 4-bit quantization (QLoRA).
*   **LoRA Fine-tuning:** Allows efficient adaptation of a pre-trained base model using Low-Rank Adaptation.

## Prerequisites

*   **Python:** Version 3.10 is recommended (as seen in the original `README.md` setup).
*   **GPU:** A CUDA-enabled GPU is highly recommended for both training and efficient inference (especially with 4-bit loading). Training typically requires substantial VRAM. CPU inference is possible but will be very slow.
*   **CUDA Toolkit:** Ensure you have a compatible CUDA Toolkit installed if using a GPU.
*   **Git:** For cloning the repository.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/freds0/Orpheus-TTS-ptbr
    cd Orpheus-TTS-ptbr
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    # Using venv
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`

    # Or using conda
    # conda create -n orpheus-tts python=3.10
    # conda activate orpheus-tts
    ```

3.  **Install PyTorch:**
    Install the appropriate PyTorch version for your system (CPU or specific CUDA version). Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command. Example for CUDA 12.1:
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **Install dependencies:**
    Install the required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This includes `unsloth`, `snac`, `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`, etc.*

5.  **(Optional) Install `hf_transfer` for faster Hugging Face downloads:**
    ```bash
    pip install hf_transfer
    export HF_HUB_ENABLE_HF_TRANSFER=1
    ```

## Usage

### 1. Downloading Pre-trained Models/Checkpoints (Optional)

The project uses a base model (e.g., `unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit`) which is typically downloaded automatically during training or inference initiation by the `transformers` or `unsloth` library.

However, if you have fine-tuned adapters hosted separately on the Hugging Face Hub (like the example `freds0/orpheus-3b-0.1-finetuned-ptbr`), you can use the provided script to download them:

```bash
python download_checkpoints_huggigface.py \
    --repo_id freds0/orpheus-3b-0.1-finetuned-ptbr \
    --local_dir ./downloaded_adapters
    # Optional: --token YOUR_HF_TOKEN (if private repo)
```

This script downloads the contents of the specified Hugging Face repository to a local directory.

### 2. Running Inference (Generating Speech)

The `inference.py` script generates audio files from text sentences using a fine-tuned checkpoint (LoRA adapters).

**Steps:**

1.  **Prepare an input text file:** Create a `.txt` file (e.g., `sentences.txt`) with one sentence per line that you want to convert to speech.
    ```txt
    Olá mundo!
    Este é um teste do sistema de conversão de texto em fala.
    Espero que a qualidade do áudio seja boa.
    ```

2.  **Run the inference script:**
    Use the `run_inference.sh` script or execute `inference.py` directly, providing the necessary paths.

    *Using `run_inference.sh` (modify paths as needed):*
    ```bash
    # Edit run_inference.sh first if your paths/settings are different
    bash run_inference.sh
    ```

    *Running `inference.py` directly:*
    ```bash
    # Define where your fine-tuned adapter checkpoint is located
    CHECKPOINT_PATH="./outputs/checkpoint-16000" # Or the path from the download script

    # Define the base model used during fine-tuning
    BASE_MODEL="unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # Or your specific base model

    # Define your input text file
    INPUT_FILE="./sentences.txt"

    # Define where to save the generated .wav files
    OUTPUT_DIR="./generated_audio_batch"

    python inference.py \
        --checkpoint_path ${CHECKPOINT_PATH} \
        --base_model ${BASE_MODEL} \
        --input_txt ${INPUT_FILE} \
        --output_dir ${OUTPUT_DIR} \
        --device cuda \
        # --- Optional arguments ---
        # --max_new_tokens 2500     # Max audio tokens to generate per sentence
        # --temperature 0.7         # Sampling temperature
        # --top_p 0.95              # Nucleus sampling p
        # --no-load_in_4bit         # Disable 4-bit loading (uses more VRAM/RAM)
    ```

3.  **Check the output:** The generated audio files (e.g., `0001.wav`, `0002.wav`, ...) will be saved in the specified `--output_dir`.

### 3. Running Training (Fine-tuning)

The `train.py` script fine-tunes a base Orpheus model on a custom dataset using LoRA.

**Steps:**

1.  **Prepare your dataset:** Ensure your dataset is compatible (e.g., hosted on Hugging Face Hub) and contains 'audio' and 'text' columns. The script expects audio data that can be processed by SNAC.
2.  **Configure the training script:** Modify `run_train.sh` or prepare your command-line arguments for `train.py`.
3.  **Run the training script:**

    *Using `run_train.sh` (modify paths and parameters as needed):*
    ```bash
    # Edit run_train.sh first to set your dataset, output dir, etc.
    bash run_train.sh
    ```

    *Running `train.py` directly (example based on `run_train.sh`):*
    ```bash
    # Define the base model to fine-tune
    BASE_MODEL="unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # Or your chosen base

    # Define the dataset on Hugging Face Hub or local path
    DATASET_NAME="freds0/BRSpeech-TTS-Leni"

    # Define where to save checkpoints and logs
    OUTPUT_DIR="./checkpoints_orpheus_ptbr_finetune"

    # Optional: Path to resume training from a specific checkpoint
    # RESUME_CHECKPOINT="./checkpoints_orpheus_ptbr_finetune/checkpoint-5000"

    python train.py \
        --base_model_name ${BASE_MODEL} \
        --dataset_name ${DATASET_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --max_steps 100000 \
        --save_steps 2500 \
        --logging_steps 10 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-4 \
        --lr_scheduler_type "cosine" \
        --warmup_steps 100 \
        --max_seq_length 2048 \
        --lora_r 64 \
        --lora_alpha 64 \
        --num_cpus 4 \ # Adjust based on your system
        --num_amostras -1 \ # -1 uses all samples, or set a specific number
        # --resume_from_checkpoint ${RESUME_CHECKPOINT} # Uncomment to resume
        # Add other arguments like --max_audio_duration if needed
    ```

4.  **Monitor training:** Training progress and logs (including TensorBoard logs if `tensorboard` is installed) will be saved in the `--output_dir`. Checkpoints (containing LoRA adapters) will be saved in subdirectories like `checkpoint-2500`, `checkpoint-5000`, etc.

## Dependencies

Key Python libraries used:

*   `torch`: Deep learning framework.
*   `unsloth`: For efficient LoRA training and inference.
*   `snac`: Neural audio codec.
*   `transformers`: Hugging Face library for models and tokenizers.
*   `datasets`: Hugging Face library for dataset loading and processing.
*   `peft`: Hugging Face Parameter-Efficient Fine-Tuning library.
*   `bitsandbytes`: For 4-bit quantization.
*   `accelerate`: For distributed training and mixed precision.
*   `soundfile`: For saving audio files.
*   `huggingface_hub`: For interacting with the Hugging Face Hub.

See `requirements.txt` for the full list.

