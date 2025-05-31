# Iris ML Pipeline with DVC (Week 2 Assignment)

This repository trains a Random Forest model on the Iris dataset and tracks both the dataset and model versions using **DVC** and **Git**.

We simulate two data versions:
- **v1.0**: 100 rows of `iris.csv`
- **v2.0**: Full 150 rows

---

## 🧰 Requirements

- Python 3.7+
- Git
- [DVC](https://dvc.org)
- `scikit-learn`, `pandas`, `joblib`

---

## 📦 Installation

```bash
git clone https://github.com/Mlfords123/week2.git
cd week2

python3 -m venv .env
source .env/bin/activate

pip install -r requirements.txt
pip install dvc
