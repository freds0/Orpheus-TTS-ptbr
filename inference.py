# -*- coding: utf-8 -*-
import torch
import argparse
import os
import soundfile as sf
import torchaudio.transforms as T
from unsloth import FastLanguageModel
from snac import SNAC
from peft import PeftModel
from transformers import AutoTokenizer
import locale
import numpy as np
import time # Added for timing

# Ensure UTF-8 encoding is preferred (useful in some environments)
try:
    locale.getpreferredencoding = lambda: "UTF-8"
except:
    pass

# --- Configuration Constants ---
# (Keep your existing constants: TOKENISER_LENGTH, special tokens, SNAC offsets, etc.)
TOKENISER_LENGTH = 128256
START_OF_TEXT = 128000
END_OF_TEXT = 128009
START_OF_SPEECH = TOKENISER_LENGTH + 1   # 128257
END_OF_SPEECH = TOKENISER_LENGTH + 2     # 128258
START_OF_HUMAN = TOKENISER_LENGTH + 3    # 128259
END_OF_HUMAN = TOKENISER_LENGTH + 4      # 128260
START_OF_AI = TOKENISER_LENGTH + 5       # 128261
END_OF_AI = TOKENISER_LENGTH + 6         # 128262
PAD_TOKEN = TOKENISER_LENGTH + 7         # 128263
AUDIO_TOKENS_START = TOKENISER_LENGTH + 10 # 128266

SNAC_AUDIO_TOKEN_OFFSET = AUDIO_TOKENS_START
SNAC_TARGET_SR = 24000
SNAC_MODEL_REPO = "hubertsiuzdak/snac_24khz"
SNAC_LAYER_OFFSETS_IN_CODE = [
    0, 1 * 4096, 2 * 4096, 3 * 4096, 4 * 4096, 5 * 4096, 6 * 4096
]
# --- Function Definitions ---
# (Keep your existing functions: load_models, format_input, generate_audio_tokens,
#  extract_and_clean_codes, redistribute_codes_for_snac, decode_audio, save_audio)
# Make sure they are defined here...

def load_models(checkpoint_path, base_model_name, load_in_4bit=True, device="cuda"):
    """Loads the base LLM, applies trained LoRA adapters, and loads SNAC."""
    print(f"Loading base model '{base_model_name}'...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_4bit = load_in_4bit and device == "cuda"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name, max_seq_length=2048, dtype=dtype if device == "cuda" else torch.float32, load_in_4bit=use_4bit,
    )
    print(f"Base model '{base_model_name}' loaded.")
    print(f"Loading LoRA adapter weights from checkpoint '{checkpoint_path}'...")
    try:
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
        print("Successfully loaded PEFT adapters onto the base model.")
    except Exception as e:
        print(f"\n!!! ERROR loading PEFT adapters from '{checkpoint_path}': {e}")
        raise
    print("Preparing model for inference with Unsloth optimizations...")
    FastLanguageModel.for_inference(model)
    model.eval()
    print(f"Loading SNAC model '{SNAC_MODEL_REPO}'...")
    snac_model = SNAC.from_pretrained(SNAC_MODEL_REPO)
    snac_model = snac_model.to(device)
    snac_model.eval()
    print("Models loaded successfully.")
    return model, tokenizer, snac_model

def format_input(text, tokenizer):
    """Formats the input text with special tokens."""
    # SOH + SOT + text + EOT + EOH + SOA + SOS
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids_list = (
        [START_OF_HUMAN] + [START_OF_TEXT] + text_ids + [END_OF_TEXT] +
        [END_OF_HUMAN] + [START_OF_AI] + [START_OF_SPEECH]
    )
    return torch.tensor([input_ids_list], dtype=torch.int64)

def generate_audio_tokens(model, input_ids, tokenizer, device, max_new_tokens=1500, temperature=0.7, top_p=0.95):
    """Generates audio tokens using the LLM."""
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    eos_token_ids = [END_OF_SPEECH]
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id != END_OF_SPEECH:
        eos_token_ids.append(tokenizer.eos_token_id)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=top_p, repetition_penalty=1.1,
            num_return_sequences=1, eos_token_id=eos_token_ids, pad_token_id=PAD_TOKEN, use_cache=True
        )
    return generated_ids

def extract_and_clean_codes(generated_ids, input_length, tokenizer):
    """Extracts relevant audio codes from the generated sequence."""
    generated_seq = generated_ids[0].tolist()
    generated_only_tokens = generated_seq[input_length:]
    codes_list = []
    for token in generated_only_tokens:
        if token == END_OF_SPEECH or token == tokenizer.eos_token_id:
            break
        if token >= SNAC_AUDIO_TOKEN_OFFSET and token < SNAC_AUDIO_TOKEN_OFFSET + (7 * 4096):
            codes_list.append(token - SNAC_AUDIO_TOKEN_OFFSET) # Remove base offset

    original_length = len(codes_list)
    remainder = original_length % 7
    if remainder != 0:
        codes_list = codes_list[:-remainder]
        # print(f"Warning: Code list length ({original_length}) not divisible by 7. Trimmed last {remainder} tokens.")

    if not codes_list:
        # print("Warning: No valid audio codes extracted after cleaning.")
        return []
    return codes_list

def redistribute_codes_for_snac(cleaned_code_list, device):
    """Redistributes the flat list of cleaned codes into SNAC's layered format."""
    if not cleaned_code_list or len(cleaned_code_list) % 7 != 0:
        print("Error: Invalid code list for SNAC redistribution.")
        return None
    num_frames = len(cleaned_code_list) // 7
    layer_1, layer_2, layer_3 = [], [], []
    for i in range(num_frames):
        base_idx = 7 * i
        layer_1.append(cleaned_code_list[base_idx]     - SNAC_LAYER_OFFSETS_IN_CODE[0])
        layer_2.append(cleaned_code_list[base_idx + 1] - SNAC_LAYER_OFFSETS_IN_CODE[1])
        layer_3.append(cleaned_code_list[base_idx + 2] - SNAC_LAYER_OFFSETS_IN_CODE[2])
        layer_3.append(cleaned_code_list[base_idx + 3] - SNAC_LAYER_OFFSETS_IN_CODE[3])
        layer_2.append(cleaned_code_list[base_idx + 4] - SNAC_LAYER_OFFSETS_IN_CODE[4])
        layer_3.append(cleaned_code_list[base_idx + 5] - SNAC_LAYER_OFFSETS_IN_CODE[5])
        layer_3.append(cleaned_code_list[base_idx + 6] - SNAC_LAYER_OFFSETS_IN_CODE[6])
    try:
        codes_layered = [
            torch.tensor(layer_1, dtype=torch.long).unsqueeze(0),
            torch.tensor(layer_2, dtype=torch.long).unsqueeze(0),
            torch.tensor(layer_3, dtype=torch.long).unsqueeze(0),
        ]
        codes_layered = [c.to(device) for c in codes_layered]
        return codes_layered
    except Exception as e:
        print(f"Error creating layered code tensors: {e}")
        return None

def decode_audio(snac_model, codes_layered, device):
    """Decodes the layered codes into an audio waveform using SNAC."""
    if codes_layered is None: return None
    snac_model.eval()
    with torch.inference_mode():
        audio_waveform = snac_model.decode(codes_layered)
    return audio_waveform.detach().cpu().squeeze()

def save_audio(waveform, filepath, sample_rate=SNAC_TARGET_SR):
    """Saves the audio waveform to a WAV file."""
    if waveform is None or waveform.numel() == 0:
        print(f"Error: No audio waveform generated for '{filepath}'. Skipping save.")
        return False # Indicate failure
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
             # This should be created outside the loop, but double-check just in case
             os.makedirs(output_dir, exist_ok=True)

        waveform_np = waveform.numpy().astype(np.float32)
        sf.write(filepath, waveform_np, sample_rate)
        # print(f"Audio saved successfully to '{filepath}'.")
        return True # Indicate success
    except Exception as e:
        print(f"Error saving audio file '{filepath}': {e}")
        return False # Indicate failure

# --- Main Execution Block ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate audio for multiple sentences using a fine-tuned Orpheus model checkpoint.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the training checkpoint directory (e.g., './outputs/checkpoint-1000').")
    parser.add_argument("--base_model", type=str, default="unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit", help="Name of the base model used during fine-tuning.")
    parser.add_argument("--input_txt", type=str, required=True, help="Path to the input text file (.txt), one sentence per line.")
    # Changed output_wav to output_dir
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory where generated audio files (.wav) will be saved.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Maximum number of new tokens (audio codes) to generate per sentence.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling top-p value.")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True, help="Load base model in 4-bit. Use --no-load_in_4bit for full precision or CPU.")

    args = parser.parse_args()

    # --- Validation ---
    if not os.path.isdir(args.checkpoint_path):
        print(f"Error: Checkpoint path '{args.checkpoint_path}' is not a valid directory.")
        exit(1)
    # Check for adapter files
    adapter_config_path = os.path.join(args.checkpoint_path, "adapter_config.json")
    adapter_model_path_st = os.path.join(args.checkpoint_path, "adapter_model.safetensors")
    adapter_model_path_bin = os.path.join(args.checkpoint_path, "adapter_model.bin")
    if not os.path.exists(adapter_config_path) or \
       not (os.path.exists(adapter_model_path_st) or os.path.exists(adapter_model_path_bin)):
         print(f"Error: Cannot find adapter config or weights in '{args.checkpoint_path}'.")
         exit(1)

    if not os.path.exists(args.input_txt):
        print(f"Error: Input text file '{args.input_txt}' not found.")
        exit(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Switching to CPU.")
        args.device = "cpu"

    # --- Create Output Directory ---
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: '{args.output_dir}'")
    except OSError as e:
        print(f"Error creating output directory '{args.output_dir}': {e}")
        exit(1)

    # --- Main Process ---
    model = None
    tokenizer = None
    snac_model = None
    sentences = []
    total_start_time = time.time()

    try:
        # --- Load Models (once) ---
        print("Loading models...")
        model_load_start = time.time()
        model, tokenizer, snac_model = load_models(args.checkpoint_path, args.base_model, args.load_in_4bit, args.device)
        print(f"Models loaded in {time.time() - model_load_start:.2f} seconds.")

        # --- Read Sentences ---
        print(f"Reading sentences from '{args.input_txt}'...")
        with open(args.input_txt, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()] # Read non-empty lines

        if not sentences:
            print("Error: Input text file contains no non-empty lines.")
            exit(1)

        print(f"Found {len(sentences)} sentences to synthesize.")
        print("-" * 30)

        # --- Process Each Sentence ---
        num_successful = 0
        num_failed = 0
        for i, sentence in enumerate(sentences):
            sentence_start_time = time.time()
            print(f"Processing sentence {i+1}/{len(sentences)}: \"{sentence[:70]}{'...' if len(sentence) > 70 else ''}\"")
            output_filename = f"{i+1:04d}.wav" # e.g., 0001.wav, 0002.wav
            output_filepath = os.path.join(args.output_dir, output_filename)

            try:
                # 1. Format input
                input_ids = format_input(sentence, tokenizer)
                input_length = input_ids.shape[1]

                # 2. Generate audio tokens
                gen_start = time.time()
                generated_ids = generate_audio_tokens(model, input_ids, tokenizer, args.device, args.max_new_tokens, args.temperature, args.top_p)
                # print(f"  Token generation took: {time.time() - gen_start:.2f}s")


                # 3. Extract and clean codes
                cleaned_codes = extract_and_clean_codes(generated_ids, input_length, tokenizer)

                if not cleaned_codes:
                    print("  Warning: No valid audio codes extracted for this sentence. Skipping.")
                    num_failed += 1
                    continue # Skip to the next sentence

                # 4. Redistribute codes
                codes_layered = redistribute_codes_for_snac(cleaned_codes, args.device)
                if codes_layered is None:
                     print("  Warning: Failed to redistribute codes. Skipping.")
                     num_failed += 1
                     continue

                # 5. Decode audio
                decode_start = time.time()
                waveform = decode_audio(snac_model, codes_layered, args.device)
                # print(f"  Audio decoding took: {time.time() - decode_start:.2f}s")


                # 6. Save audio
                save_start = time.time()
                if save_audio(waveform, output_filepath, sample_rate=SNAC_TARGET_SR):
                    num_successful += 1
                    # print(f"  Audio saved to {output_filepath} in {time.time() - save_start:.2f}s")
                else:
                    num_failed += 1
                    print(f"  Failed to save audio for sentence {i+1}.")

                print(f"  Sentence {i+1} processed in {time.time() - sentence_start_time:.2f} seconds.")


            except Exception as e_sentence:
                num_failed += 1
                print(f"\n--- Error processing sentence {i+1} ---")
                print(f"Sentence: {sentence}")
                import traceback
                traceback.print_exc()
                print(f"Error details: {e_sentence}")
                print("------------------------------------")
                print("Attempting to continue with the next sentence...")

            print("-" * 20) # Separator between sentences

    except Exception as e_main:
        print(f"\n--- A critical error occurred during setup or processing ---")
        import traceback
        traceback.print_exc()
        print(f"Error details: {e_main}")
        print("------------------------------------")
        exit(1)
    finally:
        # Clean up GPU memory (optional, but good practice)
        del model
        del tokenizer
        del snac_model
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    total_time = time.time() - total_start_time
    print("\n" + "=" * 30)
    print("Batch Inference Summary:")
    print(f"  Total sentences processed: {len(sentences)}")
    print(f"  Successfully synthesized:  {num_successful}")
    print(f"  Failed:                    {num_failed}")
    print(f"  Total processing time:     {total_time:.2f} seconds")
    print(f"  Average time per sentence: {total_time / len(sentences):.2f} seconds (if all processed)")
    print(f"  Generated audio saved in directory: '{args.output_dir}'")
    print("=" * 30)

