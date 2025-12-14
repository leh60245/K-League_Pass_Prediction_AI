# K-League Final Pass Prediction Toolkit

## Quick Start
```bash
python -m pip install -r requirements.txt
python eda_skeleton.py --data-dir data
python feature_engineering.py --data-dir data --balance-strategy type_weight
python feature_report.py --input artifacts/features/train_features.parquet --output artifacts/reports
```
