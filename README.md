

The LLM-assisted fairness testing framework for ML models. This guide covers running
the tool on biased dataset ` with protected attribute marital status bias.

---

## Project Structure

```
agentic_fairness/
├── fairagent.py               ← main framework (do not edit)
├── run.py                     ← entry point (edit settings here)
├── bank-small.csv            
├── bank-small-biased.csv      ← biased dataset (the biased bank dataset)
└── fairagent_results/         ← output folder (auto-created)
    ├── discriminatory_pairs.csv
    ├── test_dataset_model_ready.csv
    ├── fairness_report.txt

```

---

## Requirements

```bash
pip install openai scikit-learn shap matplotlib seaborn pandas numpy
```

---

## Step 1 — Select biased Dataset



Output: `bank-small-biased.csv`

---

## Step 2 — Configure run.py

Open `run.py` and set your API key and dataset path:

```python
agent = FairAgent(
    dataset_path      = "bank-small-biased.csv",  # biased dataset
    target_col        = "y",
    api_key           = "sk-...",                 # your OpenAI key

    # Speed settings (approx 5 minutes this is done so that user can see a demo run for small scale dataset)
    n_seeds           = 5,     # seeds per iteration
    n_counterfactuals = 3,     # counterfactuals per seed
    max_retries       = 0,     # retries on failure
    n_iterations      = 1,     # number of iterations
    fairness_threshold = 0.005,
)
```

---

## Step 3 — Set API Key

```powershell
$env:OPENAI_API_KEY = "sk-your-key-here"
```

---

## Step 4 — Run FairAgent

```powershell
python run.py
```

**Expected runtime:** ~5 minutes
**Expected cost:** ~$0.10

---

## Expected Output

```
FairAgent: Bank Marketing ML Fairness Testing
Dataset:   4,521 customers
Protected: ['age', 'marital']

Measuring fairness...
  marital: SPD=+0.350 [SEVERE] BIASED

Global Search: generating 5 seeds...
Local Search (G=found, F=not found, U=invalid)
  G=15 F=30 U=0 cost=$0.08

Discriminatory pairs found: 15
Results saved to: fairagent_results/
```

---

## Dataset Details

| Property           | Value                          |
|--------------------|-------------------------------|
| File               | bank-small.csv                 |
| Rows               | 4,521 customers                |
| Columns            | 17 (16 features + target)      |
| Target             | y (yes/no subscription)        |
| Separator          | semicolon (;)                  |
| Header             | No (columns auto-assigned)     |
| Protected attrs    | age, marital                   |
| Marital values     | married, single, divorced      |
| Privileged group   | married                        |
| Unprivileged group | single                         |

---

## Fairness Metrics Explained

| Metric | Formula | Fair Value |
|--------|---------|------------|
| SPD    | P(Y=1\|married) - P(Y=1\|single) | 0.0 |



```

---

## Speed vs Quality Settings

| Setting           | Fast (~5 min) | Balanced (~15 min) | Full (~48 min) |
|-------------------|--------------|-------------------|---------------|
| n_seeds           | 5            | 10                | 20            |
| n_counterfactuals | 3            | 5                 | 10            |
| max_retries       | 0            | 1                 | 2             |
| n_iterations      | 1            | 2                 | 3             |
| Est. cost         | ~$0.10       | ~$0.50            | ~$1.65        |
| Pairs found       | ~15          | ~100              | ~309          |

---

## Output Files

| File | Description |
|------|-------------|
| `discriminatory_pairs.csv` | Full pairs with metadata and predictions |
| `test_dataset_model_ready.csv` | Encoded pairs ready for retraining |
| `fairness_report.txt` | Detailed SPD/EOD/DI/AOD report |


---

## How It Works

```
1. PERCEIVE  → measure SPD on model
2. REASON    → LLM global search (0-shot, SHAP-guided)
3. ACT       → LLM local search (counterfactual generation on seeds)
4. EVALUATE  → f(x) ≠ f(x') → discrimination found
5. REFLECT   → retry on failure (becuase of ML Randomness)
6. REPEAT    → n_iterations times
```

---



---


