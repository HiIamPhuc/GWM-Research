#!/bin/bash
# Train GWM-RNN with Text Finetuning

python ../utils/preprocess_data.py \
    --data_dir d:/NLP research/Code/graph-world-models/GWM-Research/data/WN18RR \
    --output_dir d:/NLP research/Code/graph-world-models/GWM-Research/data/processed/wn18rr
    
python ../utils/compute_context.py \
    --data_dir d:/NLP research/Code/graph-world-models/GWM-Research/data/processed/wn18rr \
    --k 10

python ../train.py \
    --config d:/NLP research/Code/graph-world-models/GWM-Research/configs/wn18rr_finetune.yaml
