# Iris ML Pipeline with DVC

This project trains a Random Forest model on the Iris dataset and tracks the dataset and model using DVC.

## 📁 Structure

- `data/iris.csv` — input dataset
- `models/model.joblib` — output model
- `train.py` — training script
- `dvc` — handles data and model versioning

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/iris-pipeline.git
cd iris-pipeline

# Set up environment
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt

# Get data
dvc pull

# Train
python train.py

# Push model to DVC remote
dvc add models/model.joblib
dvc push
