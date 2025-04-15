python train.py \
    --dataset_name freds0/BRSpeech-TTS-Leni \
    --output_dir ./checkpoints_orpheus_ptbr_15-04-2024 \
    --max_steps 100000 \
    --save_steps 2500 \
    --num_cpus 2 \
    --num_amostras -1 \
    --resume_from_checkpoint ./checkpoints_orpheus_ptbr/checkpoint-16000/

