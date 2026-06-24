# Polarity-aware PA-CCS on ToxiGen: pair each toxic text with an LLM-generated benign rewrite
# (the opposite, non-toxic claim about the same target group) so the full PA-CCS metrics
# (polar_consistency / contradiction_index) are defined.
#
# Requires gated ToxiGen access (`hf auth login`) and an OpenRouter key for the rewrites.
# Output lives under data/toxigen/ which is gitignored -- do not commit it to this public repo.
export OPENROUTER_API_KEY=sk-or-...

# Step 1: download ToxiGen, keep the toxic rows, and generate a benign opposite for each.
# The rewrite prompt reverses the claim (same group/topic), mirrors structure, stays under
# ~25 words, and adds no new facts. Failed/unchanged rewrites are dropped.
latent-align prepare-toxigen \
  --with-negations \
  --config annotated \
  --split test \
  --toxic-threshold 3.0 \
  --gen-model openai/gpt-4o-mini \
  --output data/toxigen/raw/toxigen_annotated_test_paired.csv

# Step 2: PA-CCS on the opposite-polarity pairs (paired format).
latent-align run \
  --dataset data/toxigen/raw/toxigen_annotated_test_paired.csv \
  --dataset-format paired \
  --model allenai/OLMo-1B-hf \
  --model-kind decoder \
  --strategy last-token \
  --output-dir runs/olmo_1b_toxigen_paired \
  --normalizing l2,median
