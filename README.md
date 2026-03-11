# nl2vis-context

Minimal reproducible project for studying context engineering in natural-language-to-visualization generation on `nvBench`.

## Included

- `nvBench.json`: primary benchmark file
- `database/`: supporting SQLite databases and schema resources
- `scripts/context_engineering_study.py`: main training and evaluation pipeline
- `scripts/build_context_report.py`: report builder
- `requirements.txt`: Python dependencies

## Not Included

- Any generated artifacts or experiment outputs
- Temporary notes
- Smoke-test folders

Users can rerun the experiments locally and compare the outputs with the results described in the paper/report.

## Environment

- Python 3.11+ recommended
- CUDA-enabled PyTorch recommended for training

## Install

```bash
pip install -r requirements.txt
```

## Main Commands

Run the main `t5-small` context study:

```bash
python scripts/context_engineering_study.py
```

Run the stronger `t5-small` schema fine-tuning:

```bash
python scripts/context_engineering_study.py --variants schema --epochs 4 --batch-size 8 --train-all-queries --artifacts-dir artifacts_context_study_schema_ft
```

Run LongT5 on baseline and schema:

```bash
python scripts/context_engineering_study.py --model-name google/long-t5-tglobal-base --variants baseline schema --epochs 3 --batch-size 4 --lr 1e-4 --max-source-len 1536 --split-source-dir artifacts_context_study --train-all-queries --use-amp --artifacts-dir artifacts_context_study_longt5_ft
```

Build a summary report from generated artifacts:

```bash
python scripts/build_context_report.py
```

## Notes

- Generated outputs are written into artifact folders such as `artifacts_context_study/`.
- These artifact folders are excluded by `.gitignore`.
- The first run may take longer because database metadata is built from the SQLite files.
