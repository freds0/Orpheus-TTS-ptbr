import argparse
import os
import locale
import torch
import torchaudio.transforms as T
from datasets import load_dataset
# Import AutoTokenizer
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from unsloth import FastLanguageModel
from snac import SNAC
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

locale.getpreferredencoding = lambda: "UTF-8"

# --- Function Definitions ---
# (Keep your existing functions: is_duration_ok, converter_item, tokenise_audio,
#  add_codes, remove_duplicate_frames, create_input_ids)
# Certifique-se de que elas estejam definidas aqui...

def is_duration_ok(example, max_duration=10.0):
    """Checks if the audio duration is less than or equal to max_duration."""
    audio_info = example.get('audio')
    if not audio_info or 'array' not in audio_info or 'sampling_rate' not in audio_info:
        return False
    sampling_rate = audio_info['sampling_rate']
    if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
        return False
    audio_array = audio_info['array']
    if not isinstance(audio_array, (np.ndarray, list)) or not hasattr(audio_array, '__len__') or len(audio_array) == 0:
         return False
    num_samples = len(audio_array)
    duration = num_samples / sampling_rate
    return duration <= max_duration

def converter_item(item):
    """Converts a dataset item to the desired format."""
    text = item['text']
    audio_data = item['audio']['array']
    sampling_rate = item['audio']['sampling_rate']
    return {'audio': {'array': audio_data, 'sampling_rate': sampling_rate}, 'text': text}

def tokenise_audio(waveform, ds_sample_rate, snac_model, device="cuda"):
    """Tokenises audio waveform using SNAC."""
    try:
        if not isinstance(waveform, np.ndarray):
            waveform = np.array(waveform)
        waveform = torch.from_numpy(waveform).unsqueeze(0).to(dtype=torch.float32)
        if ds_sample_rate != 24000:
            resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=24000)
            waveform = resample_transform(waveform)
        waveform = waveform.unsqueeze(0).to(device)
        with torch.inference_mode():
            codes = snac_model.encode(waveform)
        all_codes = []
        offsets = [0, 4096, 2*4096, 3*4096, 4*4096, 5*4096, 6*4096]
        base_offset = 128266
        num_frames = codes[0].shape[1]
        for i in range(num_frames):
            all_codes.append(codes[0][0, i].item() + base_offset + offsets[0])
            all_codes.append(codes[1][0, 2*i].item() + base_offset + offsets[1])
            all_codes.append(codes[2][0, 4*i].item() + base_offset + offsets[2])
            all_codes.append(codes[2][0, 4*i + 1].item() + base_offset + offsets[3])
            all_codes.append(codes[1][0, 2*i + 1].item() + base_offset + offsets[4])
            all_codes.append(codes[2][0, 4*i + 2].item() + base_offset + offsets[5])
            all_codes.append(codes[2][0, 4*i + 3].item() + base_offset + offsets[6])
        return all_codes
    except Exception as e:
        logger.error(f"Error during audio tokenization: {e}", exc_info=True)
        return None

def add_codes(example, ds_sample_rate, snac_model):
    """Adds SNAC codes to the example."""
    codes_list = None
    try:
        answer_audio = example.get("audio")
        if answer_audio and "array" in answer_audio and "sampling_rate" in answer_audio:
            audio_array = answer_audio["array"]
            current_sample_rate = answer_audio.get("sampling_rate", ds_sample_rate)
            codes_list = tokenise_audio(audio_array, current_sample_rate, snac_model)
    except Exception as e:
        logger.warning(f"Skipping row due to error in add_codes: {e}")
    example["codes_list"] = codes_list
    return example

def remove_duplicate_frames(example):
    """Removes consecutive duplicate SNAC frames."""
    codes_list = example.get("codes_list")
    if not codes_list: return example
    if len(codes_list) % 7 != 0:
        logger.warning(f"Input list length {len(codes_list)} not divisible by 7. Skipping duplicate removal.")
        return example
    if len(codes_list) < 7:
        return example
    result = codes_list[:7]
    for i in range(7, len(codes_list), 7):
        current_frame = codes_list[i : i+7]
        previous_frame = result[-7:]
        if current_frame != previous_frame:
            result.extend(current_frame)
    example["codes_list"] = result
    return example

def create_input_ids(example, tokenizer): # Pass tokenizer explicitly
    """Creates input_ids, labels, and attention_mask for training."""
    tokeniser_length = 128256
    start_of_text = 128000
    end_of_text = 128009
    start_of_speech = tokeniser_length + 1
    end_of_speech = tokeniser_length + 2
    start_of_human = tokeniser_length + 3
    end_of_human = tokeniser_length + 4
    start_of_ai = tokeniser_length + 5
    end_of_ai = tokeniser_length + 6

    codes_list = example.get("codes_list")
    if not codes_list:
        logger.warning("Skipping create_input_ids: codes_list is missing or None.")
        return None

    text_prompt = f"{example['source']}: {example['text']}" if "source" in example else example["text"]
    if not text_prompt:
        logger.warning("Skipping create_input_ids: text_prompt is empty.")
        return None

    # Use the provided tokenizer
    text_ids = tokenizer.encode(text_prompt, add_special_tokens=False)

    input_ids_list = (
        [start_of_human] + [start_of_text] + text_ids + [end_of_text] +
        [end_of_human] + [start_of_ai] + [start_of_speech] +
        codes_list +
        [end_of_speech] + [end_of_ai]
    )
    labels = input_ids_list[:]
    example["input_ids"] = input_ids_list
    example["labels"] = labels
    example["attention_mask"] = [1] * len(input_ids_list)
    return example


# --- Main Training Function ---
def main(args):
    logger.info("Starting training script...")
    logger.info(f"Arguments: {args}")

    # --- Tokenizer Loading (BEFORE dataset processing) ---
    logger.info("Loading ONLY the tokenizer...")
    try:
        # Load tokenizer using AutoTokenizer from the base model name
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name,
            # token = "hf_..." # Add if your base model is gated
            trust_remote_code=True # Often needed for custom tokenizers/models
        )
        # Note: Manual special token definitions are in create_input_ids.
        # Ensure padding token is set if needed by potential collators, though not explicitly used here yet.
        # if tokenizer.pad_token is None:
        #    logger.warning("Tokenizer does not have a pad token set. Setting to eos_token.")
        #    tokenizer.pad_token = tokenizer.eos_token # Common practice
        logger.info("Tokenizer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer for '{args.base_model_name}': {e}")
        raise

    # --- Dataset Loading and Preprocessing ---
    logger.info(f"Loading dataset '{args.dataset_name}'...")
    try:
        dataset = load_dataset(args.dataset_name, split='train')
    except Exception as e:
        logger.error(f"Failed to load dataset '{args.dataset_name}': {e}")
        raise

    logger.info(f"Original dataset size: {len(dataset)}")
    dataset = dataset.shuffle(seed=args.seed)

    if args.num_amostras != -1:
        logger.info(f"Selecting first {args.num_amostras} samples.")
        if args.num_amostras > len(dataset):
             logger.warning(f"Requested {args.num_amostras} samples, but dataset only has {len(dataset)}. Using all samples.")
             args.num_amostras = len(dataset)
        dataset = dataset.select(range(args.num_amostras))
        logger.info(f"Dataset size after selection: {len(dataset)}")

    logger.info("Applying basic item conversion...")
    dataset = dataset.map(converter_item, num_proc=args.num_cpus)

    max_seconds = args.max_audio_duration
    logger.info(f"Applying duration filter (max_duration = {max_seconds}s)...")
    original_count = len(dataset)
    num_cpus_filter = min(os.cpu_count(), args.num_cpus)
    dataset = dataset.filter(
        lambda example: is_duration_ok(example, max_duration=max_seconds),
        num_proc=num_cpus_filter
    )
    filtered_count = len(dataset)
    logger.info(f"Dataset size after filtering: {filtered_count} (Removed {original_count - filtered_count} samples).")
    if filtered_count == 0:
        logger.error("No samples remaining after duration filtering!")
        raise ValueError("Filtering resulted in an empty dataset.")

    # --- SNAC Tokenization (Requires SNAC model) ---
    logger.info("Loading SNAC model...")
    try:
        snac_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device {snac_device} for SNAC model.")
        snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(snac_device)
        snac_model.eval()
        logger.info("SNAC model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load SNAC model: {e}")
        raise

    ds_sample_rate = dataset[0]["audio"]["sampling_rate"]
    logger.info(f"Dataset sample rate (from first item): {ds_sample_rate}")

    logger.info("Adding SNAC codes...")
    dataset = dataset.map(
        lambda example: add_codes(example, ds_sample_rate, snac_model),
        remove_columns=["audio"]
    )
    # Release SNAC model memory if possible (optional, depends on GPU RAM)
    # del snac_model
    # torch.cuda.empty_cache()
    # logger.info("SNAC model unloaded.")


    original_count = len(dataset)
    dataset = dataset.filter(lambda x: x["codes_list"] is not None and len(x["codes_list"]) > 0)
    filtered_count = len(dataset)
    logger.info(f"Dataset size after filtering failed tokenizations: {filtered_count} (Removed {original_count - filtered_count} samples).")
    if filtered_count == 0:
        logger.error("No samples remaining after filtering failed tokenizations!")
        raise ValueError("Tokenization resulted in an empty dataset.")

    logger.info("Removing duplicate frames...")
    dataset = dataset.map(remove_duplicate_frames)

    # --- Final Input Formatting (Uses the pre-loaded tokenizer) ---
    logger.info("Creating final input format (input_ids, labels, attention_mask)...")
    dataset = dataset.map(
        lambda example: create_input_ids(example, tokenizer), # Pass the pre-loaded tokenizer
        remove_columns=[col for col in dataset.column_names if col not in ["input_ids", "labels", "attention_mask"]]
    )

    original_count = len(dataset)
    dataset = dataset.filter(lambda example: example is not None)
    filtered_count = len(dataset)
    logger.info(f"Dataset size after filtering failed formatting: {filtered_count} (Removed {original_count - filtered_count} samples).")
    if filtered_count == 0:
        logger.error("No samples remaining after final formatting!")
        raise ValueError("Final formatting resulted in an empty dataset.")

    logger.info("Dataset preprocessing complete.")
    # --- At this point, the dataset is fully processed ---

    # --- Model Loading (AFTER dataset processing) ---
    logger.info("Loading base model AFTER dataset processing...")
    dtype = None
    load_in_4bit = True
    try:
        # Load the model using FastLanguageModel. We don't strictly need the tokenizer it returns
        # because we already loaded one, but FastLanguageModel likely expects to handle both.
        # Ensure the underlying tokenizer loaded by FastLanguageModel is compatible with the one
        # we loaded earlier (should be if base_model_name is the same).
        model, loaded_tokenizer_ignore = FastLanguageModel.from_pretrained(
            model_name=args.base_model_name,
            max_seq_length=args.max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            # token = "hf_..." # Add if your base model is gated
            # trust_remote_code=True # May be needed depending on the model
        )
        # Optional: Verify loaded_tokenizer_ignore is similar to the pre-loaded tokenizer if desired
        logger.info("Base model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load base model '{args.base_model_name}': {e}")
        raise

    # --- PEFT Setup ---
    logger.info("Applying PEFT LoRA configuration...")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )
        logger.info("PEFT model created.")
        model.print_trainable_parameters()
    except Exception as e:
        logger.error(f"Failed to apply PEFT configuration: {e}")
        raise


    # --- Trainer Setup ---
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs = {"use_reentrant" : False},
        ddp_find_unused_parameters=False,
    )

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer, # Use the tokenizer loaded at the beginning
        train_dataset=dataset,
        args=training_args,
        # data_collator=... # Add if needed
        # callbacks=...     # Add if needed
    )
    logger.info("Trainer initialized.")

    # --- Training ---
    logger.info("Starting training...")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    max_memory = round(gpu_stats.total_memory / 1024**3, 3)
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory:.3f} GB.")
    logger.info(f"{start_gpu_memory:.3f} GB of memory allocated before training (potentially includes SNAC if not unloaded).")

    checkpoint_to_resume = args.resume_from_checkpoint
    if checkpoint_to_resume:
        if os.path.isdir(checkpoint_to_resume):
            logger.warning(f"Resuming training from checkpoint: {checkpoint_to_resume}")
        else:
            logger.warning(f"Attempting to resume from path: {checkpoint_to_resume}. Ensure this is a valid checkpoint directory.")
    else:
        logger.warning("Starting training from scratch (no checkpoint specified).")

    try:
        train_result = trainer.train(resume_from_checkpoint=checkpoint_to_resume)
    except ValueError as e:
         if "No valid checkpoint found" in str(e) and checkpoint_to_resume:
              logger.error(f"Could not find a valid checkpoint to resume from at '{checkpoint_to_resume}'. Starting training from scratch instead.")
              train_result = trainer.train(resume_from_checkpoint=None)
         else:
              raise

    # --- Post-Training ---
    logger.info("Training finished.")
    # trainer.save_model() # Optional: Saves final adapter
    # logger.info(f"Final model saved to {args.output_dir}")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    end_gpu_memory = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    peak_memory_during_train = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    logger.info(f"Peak allocated GPU memory: {end_gpu_memory:.3f} GB.")
    logger.info(f"Peak reserved GPU memory: {peak_memory_during_train:.3f} GB.")
    logger.info("Script finished successfully.")


# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or resume training for an Orpheus TTS model (Tokenizer loaded first).")

    # Model and LoRA args
    parser.add_argument("--base_model_name", type=str, default="unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit", help="Base model name from Hugging Face.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the model.")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank 'r'.")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout.")

    # Dataset args
    parser.add_argument("--dataset_name", type=str, default="freds0/BRSpeech-TTS-Leni", help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--num_amostras", type=int, default=-1, help="Number of dataset samples to use (-1 for all).")
    parser.add_argument("--max_audio_duration", type=float, default=10.0, help="Maximum audio duration in seconds for filtering.")

    # Preprocessing args
    parser.add_argument("--num_cpus", type=int, default=os.cpu_count() // 2 or 1, help="Number of CPUs for data processing (map/filter).")

    # Training args - Mirroring TrainingArguments where appropriate
    parser.add_argument("--output_dir", type=str, default="./outputs_tokenizer_first", help="Directory to save checkpoints and logs.") # Changed default dir slightly
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a specific checkpoint directory to resume training from.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total number of training steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save a checkpoint every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main(args)

