OUTPUT_DIRECTORY="./generated_audio_batch" # Define your desired output folder
python inference.py \
    --checkpoint_path ./outputs/checkpoint-16000 \
    --input_txt ./sentences.txt \
    --output_dir ${OUTPUT_DIRECTORY} \
    --device cuda 
    # Optional: --base_model <name>
    # Optional: --max_new_tokens 2500
    # Optional: --temperature 0.65
    # Optional: --no-load_in_4bit
