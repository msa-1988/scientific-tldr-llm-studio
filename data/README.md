# Data Notes

This project uses the public `allenai/scitldr` dataset from Hugging Face.

No dataset files are committed to Git.

Use:

```bash
python scripts/prepare_dataset.py
```

That command downloads the dataset, creates compact train / validation / test JSONL files in `data/processed/`, and writes a dataset profile to `artifacts/`.

