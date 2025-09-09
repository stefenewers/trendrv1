# Trendr — ETH Price Features & Streamlit Explorer
Trendr is a lightweight ML project for generating features and training a model on ETH-USD time series, then exploring predictions and signals in a Streamlit app.

## ✨ Key Features
- Daily ETH-USD feature pipeline (rolling stats, returns, technicals)
- Reproducible training script that outputs artifacts to artifacts/models/
- Streamlit app for visualizing features, predictions, and simple signals
- Deterministic seeds, config via .env, and clear file layout

## 🧱 Project Structure
trendr/
├─ app.py                       # Streamlit app entry
├─ eth_predictor/               # Python package code
│  ├─ __init__.py
│  ├─ config.py                 # Paths, constants, env handling
│  ├─ data.py                   # Download/load raw, split train/test
│  ├─ features.py               # Feature engineering
│  ├─ train.py                  # Train & persist model
│  ├─ predict.py                # Load model & predict helpers
│  └─ utils.py                  # Small shared utilities (seeding, logging)
├─ data/
│  ├─ raw/                      # Raw inputs (ignored by git)
│  └─ processed/                # Engineered features (.csv) (ignored/optional LFS)
├─ artifacts/
│  └─ models/                   # Saved models (.joblib) (ignored/optional LFS)
├─ notebooks/                   # Optional EDA (ignored by git)
├─ .github/workflows/ci.yml     # CI pipeline
├─ .env.example                 # Example env config
├─ requirements.txt             # Runtime deps
├─ pyproject.toml               # (Optional) tooling config
├─ .gitignore
└─ README.md

## 🚀 Quickstart
### 0) Setup
python3 -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  
cp .env.example .env  

### 1) Generate features
python -m eth_predictor.features  
# outputs e.g. data/processed/features_ETH-USD_1d.csv  

### 2) Train the model
python -m eth_predictor.train  
# outputs e.g. artifacts/models/model_ETH-USD_1d.joblib  

### 3) Run the app
streamlit run trendr/app.py  

## ⚙️ Configuration
All tunables live in eth_predictor/config.py and .env:  
- SYMBOL (default ETH-USD)  
- FREQ   (default 1d)  
- DATA_DIR, ARTIFACT_DIR  
- SEED  

See .env.example for safe defaults.

## 📦 Reproducibility
- Fixed random seed  
- Explicit requirements.txt  
- CI checks: formatting, linting, unit tests (small/fast)  

## 🧪 Tests (optional but recommended)
Place tests in tests/ and run:  
pytest -q  

## 📊 Data & Artifacts Policy
- Large CSVs & model files are not committed by default.  
- If you must track large binary artifacts, use Git LFS.  

## 🛡️ Safety & Secrets
- No API keys are required by default.  
- Never commit .env. Use .env.example for documentation.  
- Scan your history before publishing.  

## 🙌 Acknowledgements
Built with Streamlit, scikit-learn, pandas, numpy.
