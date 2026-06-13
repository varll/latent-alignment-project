# latent-alignment-project

Reusable code for running PA-CCS style latent-alignment probes on HuggingFace models.

The implementation is based on the public `SadSabrina/polarity-probing` repository:
it keeps the same core experiment shape, but moves notebooks into reusable modules and a CLI.

## Install

Create an environment with `uv` and install from `requirements.txt`:

```bash
uv venv
uv pip install -r requirements-dev.txt   # runtime + pytest/ruff (use requirements.txt for runtime only)
uv pip install -e .                       # install the latent-align CLI
```

Then run:

```bash
latent-align --help
pytest -q
ruff check .
```

(`requirements-dev.txt` pulls in `requirements.txt`, so it covers everything.)

## Included Data

Reference datasets from `polarity-probing` are vendored here:

```text
data/polarity_probing/raw/mixed_dataset.csv
data/polarity_probing/raw/not_dataset.csv
```

They use the original `statement` + first-half/second-half pairing convention. Pass
`--dataset-format polarity_raw` explicitly when using them.

## Run An Experiment

OLMo on the mixed dataset with median normalization (as in the notebook):

```bash
latent-align run \
  --dataset data/polarity_probing/raw/mixed_dataset.csv \
  --dataset-format polarity_raw \
  --model allenai/OLMo-1B-hf \
  --model-kind decoder \
  --strategy last-token \
  --output-dir runs/olmo_1b_mixed \
  --normalizing median
```

For bigger models, set `--dtype bfloat16` or `--dtype float16` if your hardware supports it.

## Reuse Extracted Hidden States

Extraction is usually the expensive part. You can save embeddings once:

```bash
latent-align extract \
  --dataset data/polarity_probing/raw/mixed_dataset.csv \
  --dataset-format polarity_raw \
  --model allenai/OLMo-1B-hf \
  --model-kind decoder \
  --output runs/olmo_mixed_embeddings.npz
```

Then train probes repeatedly:

```bash
latent-align probe \
  --dataset data/polarity_probing/raw/mixed_dataset.csv \
  --dataset-format polarity_raw \
  --embeddings runs/olmo_mixed_embeddings.npz \
  --output-dir runs/olmo_mixed_probe \
  --normalizing median
```

Outputs:

```text
ccs_summary.csv       layerwise accuracy, silhouette, PC, CI
ccs_full_results.npz  arrays such as weights, per-example PC/CI
metadata.json         split indices and probe config
embeddings.npz        only for `run`
```

## Normalization

Hidden states are normalized before the probe is trained. A normalization is fitted on
the train split (statistics such as the per-feature mean or median come from train only)
and then applied to both the train and test splits, so no test information leaks in.

Available steps:

| step     | effect                                                            |
| -------- | ----------------------------------------------------------------- |
| `mean`   | subtract the per-feature train **mean** (default)                 |
| `median` | subtract the per-feature train **median**                         |
| `l2`     | scale each row to unit L2 norm                                    |
| `raw` / `none` | no normalization                                            |

### One pipeline

A single argument is one pipeline. Combine steps with a comma or `+`; they are applied
left to right:

```bash
--normalizing median          # center on the median
--normalizing l2,median       # L2-normalize, then median-center
--normalizing l2+median       # same as above
```

`l2,median` was the strongest configuration in the original experiments.

### Several pipelines in one run

Pass more than one pipeline to train and compare them in a single invocation (mirrors the
"try different normalization strategies" sweep from the notebook). Space separates
pipelines; comma/`+` combines steps inside a pipeline:

```bash
latent-align run \
  --dataset data/polarity_probing/raw/mixed_dataset.csv \
  --dataset-format polarity_raw \
  --model allenai/OLMo-1B-hf \
  --model-kind decoder \
  --strategy last-token \
  --output-dir runs/olmo_mixed_norm_sweep \
  --normalizing mean median l2 l2,median
```

This works for both `run` and `probe`. With a single pipeline the output layout is
unchanged (files written directly to `--output-dir`). With several pipelines:

```text
<output-dir>/
  ccs_summary.csv          combined summary, one extra `normalizing` column
  norm_mean/               per-pipeline results (ccs_summary.csv, *.npz, metadata.json)
  norm_median/
  norm_l2/
  norm_l2-median/
```

The combined `ccs_summary.csv` is the quickest way to compare layerwise accuracy across
normalizations; each `norm_*/` subdirectory holds the full per-pipeline artifacts.

## Own Data

Preferred CSV/JSONL format is one row per harmful/safe pair:

```csv
positive_text,negative_text,label,pair_id
"Harmful or target-side statement","Safe or opposite-side statement",0,0
```

Run it with:

```bash
latent-align run \
  --dataset data/my_pairs.csv \
  --dataset-format paired \
  --positive-col positive_text \
  --negative-col negative_text \
  --label-col label \
  --pair-id-col pair_id \
  --model allenai/OLMo-1B-hf \
  --model-kind decoder \
  --output-dir runs/my_pairs_olmo
```

If you keep the original polarity-probing layout, put all first-side statements in the
first half of the file and matching opposite-side statements in the second half.

For CCS, the loader turns each base statement into two prompts by appending ` Yes.`
and ` No.`. Override those with `--positive-suffix` / `--negative-suffix` if you want
different answer tokens or another language.

## Notes

- `model-kind decoder` + `strategy last-token` is the default choice for Qwen, OLMo,
  Llama, Gemma, Mistral-style models.
- `model-kind encoder` + `strategy first-token` is the default choice for BERT/DeBERTa.
- The CLI defaults to all hidden-state layers. Use `--one-layer --layer-index N` for a
  faster single-layer run.
