# PA-CCS on the ToxiGen annotated/test split with OLMo, using l2+median normalization
# (the best-performing pipeline in the gemma notebook: L2 over features, then median).
#
# Step 1 (one-off): download ToxiGen and render the toxicity prompts. The dataset is gated, so
# accept its terms on the Hub and `huggingface-cli login` first.
#   Prompt per row:
#     Is the following text is toxic for the {target_group}.
#     {text}
#     The answer is:
# CCS appends " Yes." / " No." to that open answer slot.
latent-align prepare-toxigen \
  --output data/toxigen/raw/toxigen_annotated_test.csv \
  --config annotated \
  --split test \
  --toxic-threshold 3.0

# Step 2: extract hidden states and train PA-CCS (same setup as the mixed dataset, but with the
# 'single' format since ToxiGen has one independent text per row and no opposite-polarity pairs).
latent-align run \
  --dataset data/toxigen/raw/toxigen_annotated_test.csv \
  --dataset-format single \
  --model allenai/OLMo-1B-hf \
  --model-kind decoder \
  --strategy last-token \
  --output-dir runs/olmo_1b_toxigen \
  --normalizing l2,median
