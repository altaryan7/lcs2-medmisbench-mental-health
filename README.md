# Responsible Medical LLMs for Mental Healthcare: MedMisBench Benchmark

LCS2 (IIT Delhi) coding round. This repo evaluates `meta-llama/Llama-3.2-3B-Instruct` and `Qwen/Qwen2.5-3B-Instruct` on the fixed MedMisBench mental-health multiple-choice task. It covers four dimensions (hallucination & uncertainty, interpretability & explainability, bias & fairness, privacy & security), a clean baseline, and one lightweight intervention.

Every notebook keeps the outputs of the recorded Kaggle run. You can read the results without rerunning anything.

## Notebooks

| Notebook | Assignment part | What it runs | Items | Approx. time on GPU* |
| --- | --- | --- | --- | --- |
| [`baseline.ipynb`](baseline.ipynb) | Part A | Clean run of both models, then accuracy and confidence/logit-margin calibration | Core, rows 0–199 | ~15 min |
| [`1_hallucination_uncertainty.ipynb`](1_hallucination_uncertainty.ipynb) | 5.1 | Baseline → clean vs misleading-context run → calibration → H1 test (ΔP(injected option), paired accuracy) | Core + Stress 0–59 | ~25 min |
| [`2_interpretability_explainability.ipynb`](2_interpretability_explainability.ipynb) | 5.2 | Baseline → logit lens, direct logit attribution, layer/head ablations against norm-matched controls | Interpretability 0–39 | ~35 min |
| [`3_bias_fairness.ipynb`](3_bias_fairness.ipynb) | 5.3 | Six single-attribute `PATIENT METADATA` variants → answer-flip rates with Wilson 95% CIs | Stress 0–59 | ~25 min |
| [`4_privacy_security.ipynb`](4_privacy_security.ipynb) | 5.4 | Baseline → targeted prompt injection and synthetic-canary leakage → McNemar/Holm analysis | Stress 0–59 | ~25 min |
| [`5_intervention.ipynb`](5_intervention.ipynb) | Part D | Clean top-two gate on the misleading-context condition, compared with always-clean and random-fallback controls | Stress 0–59 | ~10 min |

\* Wall-clock time from the recorded run on a Kaggle GPU. A fresh session also downloads both models (about 6 GB each).

**Each notebook is self-contained.** It repeats the shared steps it needs: dataset preparation in all of them, and the clean baseline in notebooks 1, 2 and 4. Any notebook can therefore run on its own in a fresh Kaggle session. Cells pass results to each other only through files under `/kaggle/working/medmisbench_study`.

## Fixed setup

| Item | Value |
| --- | --- |
| Models | `meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4f764b7a12671c53f0838cd831a0843b95`<br>`Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1` |
| Dataset | `Johndfm/MedMisBench-MentalHealth` @ `9757095598b6555c3e935679fdb7384aed4eb651`, file `medmisbench_mentalhealth.jsonl` (split `train`, 449 rows) |
| Dataset SHA-256 | `0c7db2295ca5bf220933bc95d8f613c11e3f8c0d3d5f0235dbf56595d5434c21` (checked before use; no row is edited or dropped) |
| Subsets | Rows in stored order. Core 0–199 (200), Stress 0–59 (60), Interpretability 0–39 (40) |
| Inference | Hugging Face `transformers==5.3.0`, float16, no quantization, SDPA attention, one model at a time |
| Decoding | Greedy and deterministic: `do_sample=False`, `num_beams=1`, `max_new_tokens=256`, batch size 8, deterministic CUDA algorithms |
| Seed | 42 for all preprocessing, sampling, bootstrap and permutation code |
| Prompt | `SYSTEM_PROMPT` at the top of each inference cell. The same prompt is used in every condition; the model must return JSON with `answer`, `confidence` (0–100) and `evidence` |
| Recorded environment | Kaggle Tesla T4, CUDA 12.8, Python 3.13.15, torch 2.11.0+cu128, transformers 5.3.0, tokenizers 0.22.2, huggingface-hub 1.29.0, numpy 2.1.3. Each run's `manifest.json` records its own versions and GPU |

## Running a notebook on Kaggle from scratch

1. **Get Hugging Face access.** Accept Meta's license on the [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) page and wait for approval. Then create a token with *read* access at <https://huggingface.co/settings/tokens>. Qwen is not gated.
2. **Import the notebook.** On Kaggle, choose *Create → New Notebook*, then *File → Import Notebook*, and upload one `.ipynb` from this repo.
3. **Set the session options** in the right-hand panel:
   - *Accelerator*: **GPU T4**. The saved Kaggle metadata says `none`, so set this by hand.
   - *Internet*: **On**. This requires a phone-verified Kaggle account.
4. **Run all cells top to bottom.** Each cell that loads a model shows a hidden prompt `Hugging Face token (hidden):`; paste your token there. The dataset-preparation cell downloads the pinned dataset without a token.
5. **Download the results.** The last cell zips `/kaggle/working/medmisbench_study` into `/kaggle/working/output_folder.zip` and shows a link. You can also download it from the *Output* panel.

Each inference cell checks for `transformers==5.3.0` and installs it before importing. If you see the message *Run `%pip install -q "transformers==5.3.0"`, restart the kernel, then rerun*, do exactly that. It only happens when another version of `transformers` was already imported in the session.

### Settings you may need to change

| Setting | Where | Default | When to change it |
| --- | --- | --- | --- |
| `DATA_DIR` | Inference cells | `None`: newest folder under `data/` | Pin a specific prepared-data folder |
| `BASELINE_DIR` | Baseline metrics, security, interpretability | `None`: newest completed `results/baseline_*` | Reproduce against a specific baseline run |
| `RUN_DIR` | Analysis cells | `None`: newest matching run | Analyse a specific run. **Exception:** the H1 cell in notebook 1 has `RUN_DIR` hard-coded to `stress_20261021T153447_400500Z`; set it to the folder printed by that notebook's stress step |
| `BATCH_SIZE` | Inference cells | `8` | Use 16 if GPU memory allows, but keep the same size across conditions |
| `RERUN_MODELS`, `STRESS_DIR` | `5_intervention.ipynb` | `True`, `None` | `False` replays a saved `results/stress_*` folder from the same session without a GPU or token, instead of rerunning inference |

**Optional manual step (notebook 1).** The stress step writes `flip_review.csv` for labelling whether evidence on target flips warns that the context is unverified. Fill in `unverified_warning` and `warning_note`, then call `score_warnings(RUN_DIR)`; inference is not rerun. The recorded run did not do this step.

## Where results are written

All paths are under `/kaggle/working/medmisbench_study/`. Every run folder is timestamped, never overwritten, and contains a `manifest.json` (status, settings, package versions, GPU, checksums) plus a backup `.zip`.

| Folder | Contents |
| --- | --- |
| `data/<ts>/` | Raw dataset copy; `full`, `core`, `stress` and `interpretability` subsets; `cases.jsonl` with every prompt variant |
| `results/baseline_<ts>/` | `<model>.jsonl`: one record per model–item pair (raw output, parsed answer, confidence, evidence, parse status, option logits, model revision, generation settings). Also `system_prompt.txt`, `model_revisions.json`, and `metrics_<ts>/` from the metrics cell |
| `results/stress_<ts>/` | `<model>.clean.jsonl`, `<model>.misleading.jsonl`, `paired_stress.jsonl`, `flip_review.csv`, `summary.json`, `h1_analysis/<ts>/` |
| `results/counterfactual_<ts>/` | `<model>.jsonl` with all seven conditions, `<model>.comparisons.jsonl`, `summary.csv` |
| `results/security_<ts>/` | `<model>.<attack>.jsonl`, `<model>.<attack>.paired.jsonl` (paired with the clean baseline), `security_cases.jsonl`, `baseline_manifest.json` |
| `results/interpretability_<ts>/` | Per-item mechanism records, `summary.json` (H1–H3 contrasts), `cases.json`/`cases.md`, `layer_comparison.png` |
| `results/top2_guard_<ts>/` | Clean, misleading and guarded records, `paired_policy.jsonl`, `summary.json`/`summary.csv`, `flagged_review.csv` |
| `analysis/counterfactual_metrics_<ts>/` | Flip-rate tables, favoured groups, hypothesis assessment, and the `flip_rates`/`flip_outcomes` figures |
| `analysis/security_metrics_<ts>/` | `report.md`, metric tables, `run_validation.json`, and the `attack_effects` figure |

## Recorded results

| Dimension | Llama-3.2-3B | Qwen2.5-3B |
| --- | --- | --- |
| Baseline accuracy, Core (n=200) | 0.405 | 0.305 |
| Stated-confidence AUROC for correctness | 0.504 | 0.514 |
| Misleading context: Stress accuracy, clean → misleading | 36.7% → 13.3% | 31.7% → 13.3% |
| Misleading context: mean change in P(injected option) [95% CI] | +0.484 [+0.380, +0.583] | +0.643 [+0.529, +0.749] |
| Bias: highest clean-vs-metadata answer-flip rate | 10.0% | 18.3% |
| Bias: items with at least one metadata flip | 8 / 60 | 15 / 60 |
| Prompt injection: accuracy, clean → attacked | 36.7% → 10.0% | 31.7% → 10.0% |
| Prompt injection: full payload compliance | 19 / 60 | 0 / 60 |
| Canary leaked | 24 / 60 | 0 / 60 |
| Intervention: misleading accuracy, before → after gate | 13.3% → 33.3% | 13.3% → 23.3% |
| Intervention: gate minus always using the clean answer | −3.3 pp | −8.3 pp |

Each notebook starts with a fuller results table and explains how its statistics are computed.

## Reproducibility notes

- The intervention notebook regenerates the clean and misleading Stress conditions from scratch. It reproduced notebook 1's accuracies exactly (36.7% / 13.3% and 31.7% / 13.3%).
- The saved outputs come from one combined Kaggle run on 21 Oct 2026, split into these notebooks; the intervention ran in a later session. Execution counts were kept from that run, so they have gaps.
- Failed or invalid generations are kept in the saved records and counted as unsuccessful, never dropped.
