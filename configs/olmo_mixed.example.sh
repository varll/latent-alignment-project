# PA-CCS on the mixed dataset with OLMo, using median normalization (as in the notebook).
latent-align run \
  --dataset data/polarity_probing/raw/mixed_dataset.csv \
  --dataset-format polarity_raw \
  --model allenai/OLMo-1B-hf \
  --model-kind decoder \
  --strategy last-token \
  --output-dir runs/olmo_1b_mixed \
  --normalizing median
