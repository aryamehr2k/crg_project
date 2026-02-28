# AI Risk Intelligence System (Datathon Prototype)

## What this project does
This project builds a multi-domain **Graph Neural Network (GNN)** system that predicts and visualizes risk across:
- Public Health
- Environmental Sustainability
- Education Equity
- Civic Policy

It ingests tabular + graph data, trains a GNN model, produces risk predictions, and exposes a **Streamlit dashboard** for decision‑makers. The dashboard includes:
- AI Risk Score + severity
- Domain focus + insights
- California vs US risk comparison
- Policy impact curve (1–100% improvement)
- California district-by-month heatmap

## How it works (high level)
1. **Data prep**: synthetic data or Live Data JSON is normalized and clustered.
2. **Graph build**: nodes = regions, edges = similarity via cosine distance.
3. **GNN**: two GraphConv layers predict multi-label risks.
4. **Explainability**: top driver per risk is computed.
5. **Dashboard**: renders predictions + policy simulations.

## Project structure
```
crg_project/
├── dashboard/            # Streamlit app
├── data/
│   ├── raw/              # optional JSON queries
│   └── processed/        # processed nodes.csv
├── models/               # GNN model + train/predict
├── outputs/              # predictions.csv, metadata.json
├── utils/                # preprocessing + graph builder + Live Data client
├── main.py               # pipeline entry
└── requirements.txt
```

## Requirements
- Python 3.10+ recommended
- `pip install -r requirements.txt`

## Data sources
### Option A — Live Data JSON (recommended)
If you have a Live Data JSON file (e.g. provided by sponsor), set:
```
USE_LIVEDATA_JSON=true
LIVEDATA_JSON_PATH=/absolute/path/to/live_data_persons_history_combined.json
LIVEDATA_MAX_NODES=1000
```

### Option B — Live Data API (optional)
Set the API credentials and provide a JSON query file in `data/raw/`:
```
LIVEDATA_ORG_ID=...
LIVEDATA_API_KEY=...   # or use LIVEDATA_CLIENT_ID + LIVEDATA_CLIENT_SECRET
USE_LIVEDATA_API=true
```
Then add one of:
- `data/raw/livedata_search.json`
- `data/raw/livedata_find.json`

### Option C — Synthetic data
If no Live Data JSON/API is available, the system generates synthetic data automatically.

## How to run
From the project root:

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run the pipeline (train + predict)
```bash
python main.py
```
This writes:
- `data/processed/nodes.csv`
- `outputs/predictions.csv`

### 3) Launch the dashboard
```bash
streamlit run dashboard/app.py
```

## If you need a clean rebuild
```bash
rm data/processed/nodes.csv outputs/predictions.csv
python main.py
```

## Common notes
- **CUDA warning**: If you see `CUDA initialization` warnings, it just means GPU isn’t available. The app still runs on CPU.
- **Large JSON**: Use `LIVEDATA_MAX_NODES` to limit sample size for faster graphs.

## What to show in the terminal (for verification)
Run:
```bash
python main.py
```
You’ll see:
- Sample predictions with `region_name`
- Total regions, avg health risk
- California vs Rest of US comparison
- Top 5 regions by final score

---
If you want a different dataset schema wired in, tell me the column names and I’ll map them.
