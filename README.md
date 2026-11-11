# ML Studio: Supervised and Unsupervised Learning Playground

A full-stack machine learning playground. Upload any CSV (or pick an example dataset), let the app tell you what kind of problem it is, clean and transform the data, compare many algorithms side by side, tune them, run AutoML, look inside the results with charts and explanations, and make new predictions. The UI is glassmorphism, built with React, Tailwind CSS and Bootstrap. The backend is Python with FastAPI and scikit-learn.

**Live demo:** https://ml-studio-orpin.vercel.app
**API docs:** https://ml-studio-e6kr.onrender.com/docs

The backend runs on Render's free tier, which sleeps when idle. The first request after a quiet period can take about a minute while it wakes up. Loaded datasets and trained models live in memory and are cleared when the service restarts.

---

## Table of contents

1. [Features](#features)
2. [Tech stack](#tech-stack)
3. [Project structure](#project-structure)
4. [Setup and run](#setup-and-run)
5. [Configuration](#configuration)
6. [How to use the app](#how-to-use-the-app)
7. [Tab by tab guide](#tab-by-tab-guide)
8. [Algorithms](#algorithms)
9. [Metrics and charts](#metrics-and-charts)
10. [Dataset type detection](#dataset-type-detection)
11. [Feature engineering operations](#feature-engineering-operations)
12. [Explanations](#explanations)
13. [Performance and limits](#performance-and-limits)
14. [API reference](#api-reference)
15. [Testing and CI](#testing-and-ci)
16. [Deployment](#deployment)
17. [Design decisions](#design-decisions)
18. [Troubleshooting](#troubleshooting)
19. [Known limitations](#known-limitations)

---

## Features

**Data**
- Upload a CSV (up to 500 MB by default) or load a built-in example dataset, grouped as supervised classification, supervised regression and unsupervised clustering.
- Automatic dataset type detection (classification, regression, labelled anomaly detection, clustering, time series, text) with a reason, a suggested target column, a confidence level and a manual override.
- Dataset insights: plain-language problems and suggested fixes, such as missing values, duplicates, constant or ID-like columns, skew, outliers, correlated pairs, class imbalance and possible target leakage.
- Exploratory analysis: correlation heatmap, histograms, box plots, scatter plots with optional colouring, and summary statistics.
- Feature engineering studio with 13 operations, a step log, undo for the last 5 steps, reset, a Before vs After comparison and a cleaned-CSV download.

**Supervised learning**
- 9 regression and 9 classification algorithms, trained and compared in one click, with 3-fold cross-validation and training time.
- Option to balance classes for imbalanced data.
- Hyperparameter tuning with Optuna (random search if Optuna is not installed), with live progress and a cancel button.
- AutoML: trains every algorithm, tunes the best three, and recommends one with a plain-language explanation.
- Rich results: metric gauges, confusion matrix, classification report, ROC and precision-recall curves, one-vs-rest ROC for multiclass problems, actual vs predicted with a perfect-prediction line, residuals and feature importance.
- A "How does X work?" note under every algorithm (Learning Mode).

**Unsupervised learning**
- 5 clustering algorithms and 3 anomaly detectors.
- Automatic K selection (elbow, silhouette, Calinski-Harabasz, Davies-Bouldin, nearest cluster distance) or manual K, and the K each method suggests.
- Adjustable DBSCAN settings and expected outlier share.
- A check of anomaly detectors against a labelled outlier column (precision, recall, F1).
- PCA 2D scatter, silhouette analysis, a dendrogram for hierarchical clustering, elbow and silhouette curves, and cluster distribution.

**Prediction and explanation**
- Predict with any trained, tuned or AutoML model, with class probabilities for classifiers.
- "Why this prediction?" and "Which columns matter most" using a what-if method and permutation importance, or SHAP when it is installed.
- Download any trained model as a `.joblib` file.

**Experience and safety**
- Experiment history for the current visit, with a comparison chart and JSON and CSV export.
- Real progress bars for training, clustering, tuning and AutoML.
- Readable error messages, warnings for a wrong target type, and a friendly message when a dataset has expired.
- Memory and job limits so a public demo stays stable.
- 27 automated backend tests and a GitHub Actions workflow.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite 5, Tailwind CSS 3 (preflight disabled), Bootstrap 5.3, Recharts |
| Backend | Python, FastAPI, Uvicorn, Pydantic |
| ML and data | scikit-learn, pandas, NumPy, SciPy, Optuna, joblib |
| Optional | SHAP (explanations), pyarrow (faster CSV parsing) |
| Testing and CI | pytest, httpx, GitHub Actions |
| Hosting | Render (backend) and Vercel (frontend) |

**Prerequisites:** Python 3.10 or newer (3.12 is used on Render; the tests also pass on 3.14), Node.js 18 or newer, npm. scikit-learn 1.2 or newer is required, and the app is pinned below 1.11 because `SVC(probability=True)` is deprecated.

---

## Project structure

```
ml-studio/
├── README.md
├── .github/workflows/ci.yml   # backend tests and frontend build on every push
├── backend/
│   ├── main.py                # app, datasets, upload, EDA, training, prediction, clustering, jobs
│   ├── fe.py                  # feature engineering, undo and reset, comparison, automatic K selection
│   ├── detect.py              # dataset type detection
│   ├── insights.py            # dataset insights
│   ├── tune.py                # hyperparameter tuning, background jobs, job slots
│   ├── automl.py              # AutoML
│   ├── explain.py             # explanations (what-if, permutation, SHAP) and model download
│   ├── requirements.txt
│   ├── .env.example
│   ├── conftest.py
│   ├── pytest.ini
│   └── tests/test_api.py
└── frontend/
    ├── .env.example
    ├── index.html, package.json, vite.config.js, postcss.config.js, tailwind.config.js
    └── src/
        ├── main.jsx, index.css
        ├── App.jsx            # layout, tabs, Dataset, Explore, Supervised, Unsupervised, Predict
        ├── Engineer.jsx, Compare.jsx
        ├── Tuner.jsx, AutoML.jsx
        ├── Experiments.jsx, experiments.js
        ├── Learn.jsx, Viz.jsx
        ├── Loader.jsx, jobs.js
```

---

## Setup and run

### 1. Backend

`backend/requirements.txt`:
```
fastapi
uvicorn[standard]
python-multipart
pandas
numpy
scikit-learn<1.11
python-dotenv
optuna
```

Windows:
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

macOS and Linux:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API is at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

Optional extras:
```bash
pip install pyarrow   # faster parsing of large CSV files
pip install shap      # adds the SHAP explanation method (see Explanations)
```
If an optional package does not install on your Python version, nothing breaks. The app falls back to its built-in behaviour.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```
Open **http://localhost:5173**. Vite forwards `/api` requests to port 8000, so start the backend first.

Production build:
```bash
npm run build
```

---

## Configuration

Copy each `.env.example` to `.env`. Real `.env` files are ignored by git.

### Backend (`backend/.env`, or the Render dashboard)

| Variable | Default | Meaning |
|---|---|---|
| `MAX_UPLOAD_MB` | `500` | Largest CSV accepted (use about `25` on Render's free plan) |
| `ALLOWED_ORIGINS` | `*` | Websites allowed to call the API, comma separated. In production use your frontend URL, for example `https://your-app.vercel.app,http://localhost:5173` |
| `MAX_DATASETS` | `10` | Datasets kept in memory. The oldest are dropped first |
| `MAX_CONCURRENT_JOBS` | `2` | Training, clustering, tuning and AutoML jobs running at once, server-wide |
| `MAX_TUNE_TRIALS` | `30` | Upper limit on tuning trials per model |
| `MAX_TUNE_SECONDS` | `90` | Upper limit on tuning seconds per model |
| `TUNE_MAX_ROWS` | `10000` | Rows used inside the tuning search on big data |
| `AUTOML_MAX_ROWS` | `20000` | Training rows used for AutoML's first pass |
| `PYTHON_VERSION` | n/a | Render only. Use `3.12.7` |

### Frontend (`frontend/.env`, or the Vercel dashboard)

| Variable | Meaning |
|---|---|
| `VITE_API_URL` | Leave empty locally (the Vite proxy is used). In production set it to your backend URL with no trailing slash, for example `https://ml-studio-e6kr.onrender.com`. Vite bakes it into the build, so redeploy after changing it |

Anything starting with `VITE_` is public in the browser. Never put secrets there.

---

## How to use the app

The tabs follow a typical workflow:

**Dataset → Explore → Engineer → Before/After → Supervised or Unsupervised → Predict → History**

1. **Dataset:** upload a CSV or click an example. Read the detected type, the insights and the column table.
2. **Explore:** correlations, distributions, box plots and scatter plots.
3. **Engineer:** clean and transform the data step by step (optional).
4. **Before/After:** check what your steps changed.
5. **Supervised** (predict a column) or **Unsupervised** (find groups or outliers): pick algorithms or run AutoML.
6. **Predict:** enter values, get a prediction and see why.
7. **History:** compare the runs from this visit and export them.

A good order for messy data: remove duplicates, fill missing values, handle outliers, encode text columns, scale, then reduce dimensions if needed.

---

## Tab by tab guide

### Dataset
- Upload a `.csv`, or click an example dataset:

| Group | Dataset |
|---|---|
| Supervised: classification | Iris flowers, Wine types, Breast cancer, Credit card fraud (synthetic, about 1.5% fraud) |
| Supervised: regression | Diabetes progression |
| Unsupervised: clustering | Iris (no labels), Wine (no labels), Mall customers (synthetic), Wholesale customers (synthetic, skewed spending) |

- The **detected type badge** shows the reason, the label column, a usage hint and the confidence. If a label column exists, "Treat as unsupervised" lets you override it. Applying a feature engineering step re-runs detection and clears the override.
- **Dataset insights** lists up to 14 findings, sorted by severity, each with a suggested fix.
- The **column table** shows type, missing count and unique count.

### Explore
Correlation heatmap, histograms for up to 12 numeric columns, box plots (each on its own scale, with outlier counts), a scatter plot with optional colouring (up to 800 rows) and summary statistics.

### Engineer
Choose a step, set its options, select columns and click **Apply step**. Every step is logged. **Undo last** reverts one step (the last 5 can be undone), **Reset all** returns to the original data, and **Download current data as CSV** saves the result. Applying, undoing or resetting a step clears trained models, because they were trained on different data.

### Before/After
Rows, columns, missing cells, duplicates and column-type counts before and after, plus added, removed and type-changed columns, overlaid distributions, per-column statistics and a correlation-change heatmap. "Before" is always the data as first loaded.

### Supervised
- **Setup:** learning type, target, test size, scaling, duplicate removal, **Balance classes** (class weights for Logistic Regression, SVM, Decision Tree, Random Forest and Extra Trees), and the algorithms to compare. The type and target are pre-filled from detection, and a yellow notice offers a one-click fix if they don't match.
- **Pipelines:** each model is a scikit-learn Pipeline (imputation, optional scaling, one-hot encoding), fitted on the training split only, so nothing leaks from the test set. The split uses `random_state=42`, stratified when possible.
- **Results:** a comparison table with a metric selector and bar chart, metric gauges, and the charts listed under [Metrics and charts](#metrics-and-charts).
- **Learning Mode:** a "How does X work?" dropdown under the algorithm chips.
- **Hyperparameter tuning:** tunes the ticked algorithms with cross-validation on the training data only, then scores before and after on the untouched test set. Shows best settings and a search curve. If "after" is lower than "before", keep the defaults.
- **AutoML:** trains every algorithm with defaults, ranks them, tunes the top 3, and recommends one. A tuned version replaces the default only if it beat it in cross-validation.

### Unsupervised
- **Algorithms:** `kmeans`, `dbscan`, `agglomerative`, `gmm`, `meanshift`, `isolation_forest`, `lof`, `oneclass_svm`.
- **K selection (K-Means, Agglomerative, GMM):** let the app decide using a method you choose, or assign K manually. After every run you see the K each method suggests, and you can click one to use it.
- **Settings:** DBSCAN `eps` and minimum points, and the expected share of outliers for the anomaly detectors.
- **Label check:** for a labelled outlier column, compares the detector's flags with the labels (the rarer label value counts as the true outlier).
- **Charts:** PCA 2D view with outliers in red, silhouette analysis, a dendrogram (Agglomerative only), elbow and silhouette curves, nearest cluster distance, and cluster distribution.

### Predict
Pick any trained model for the current data, including "(tuned)" and "AutoML best" models. Enter values and click **Predict and explain**. You get the prediction, class probabilities, a "Why this prediction?" chart and an overall importance chart. **Download this model** saves it as a `.joblib` file.

### History
Every training, tuning, AutoML and clustering run since the page loaded, with settings, steps applied and a headline score. Compare runs on a chart (only runs measured with the same score share a chart) and export JSON or CSV. The list is cleared when you reload the page.

---

## Algorithms

**Regression (9):** Linear, Polynomial (degree 2), Ridge, Lasso, Decision Tree, Random Forest, KNN, SVR, Gradient Boosting.

**Classification (9):** Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Naive Bayes, Gradient Boosting, AdaBoost, Extra Trees.

**Clustering (5):** K-Means, DBSCAN, Agglomerative (hierarchical), Gaussian Mixture, Mean Shift.

**Anomaly detection (3):** Isolation Forest, Local Outlier Factor, One-Class SVM.

**Dimensionality reduction (3):** PCA, Truncated SVD, t-SNE, available as a feature engineering step. PCA also drives the 2D cluster plot.

**Reproducibility:** `random_state=42` is set on every model with a random component, on the split, on PCA and t-SNE, and on every sampling step. Linear Regression, KNN, Naive Bayes, DBSCAN, Agglomerative, Mean Shift, LOF and One-Class SVM have no random component. Random Forest, Extra Trees and KNN use all CPU cores.

---

## Metrics and charts

| Task | Metrics |
|---|---|
| Regression | MAE, MSE, RMSE, R², "within ±10% of actual", cross-validated R², training time, rows trained on |
| Classification | Accuracy, balanced accuracy, precision, recall, F1 (weighted), ROC-AUC, PR-AUC (binary), cross-validated accuracy, training time, rows trained on |
| Clustering | Silhouette, Davies-Bouldin, Calinski-Harabasz, inertia (K-Means), cluster and noise counts |
| Anomaly detection | Anomaly and normal counts, plus precision, recall and F1 against a label column when one exists |

Accuracy does not exist for regression. The closest equivalents are R² (share of variation explained) and "within ±10%".

**Charts:** confusion matrix, classification report, ROC curve, precision-recall curve, one-vs-rest ROC for up to 10 classes, actual vs predicted (the dashed line is a perfect prediction), residual distribution, feature importance, model comparison, PCA scatter, silhouette analysis, dendrogram, elbow, silhouette by K and nearest cluster distance.

**With imbalanced data**, plain accuracy is misleading. Judge by balanced accuracy, PR-AUC and the recall of the rare class in the classification report.

**Automatic K methods:** elbow (the knee of the inertia curve), silhouette (highest), Calinski-Harabasz (highest), Davies-Bouldin (lowest) and nearest cluster distance (highest ratio of the gap between the two closest cluster centres to the largest cluster radius). All are computed from a K-Means scan over K = 2 to 10.

---

## Dataset type detection

Detection runs on every upload, example load and feature engineering step.

| Detected type | Rule |
|---|---|
| Anomaly detection (labels available) | A column named like `outlier`, `anomaly`, `anomalous` or `novelty` with exactly 2 values |
| Time series | A datetime column, or a text column named like `date` or `time` that parses as dates |
| Text (NLP) | A text column with long strings (average length over 50) and mostly unique values |
| Classification | A column named like `target`, `label`, `class`, `outcome`, `species`, `diagnosis`, `survived`, `churn`, `default`, `fraud`, `price`, `salary`, `charges` or `medv` with few distinct values, or an unnamed last column with only a few whole-number values (medium confidence) |
| Regression | A named target column (same names) that is a continuous number |
| Clustering | Anything else |

Detection is a heuristic, so check the confidence label. Time series and text datasets are flagged but cannot be trained on.

---

## Feature engineering operations

| Operation | Options |
|---|---|
| Fill missing values | mean, median, mode, constant, forward fill, KNN imputation |
| Drop rows or columns with nulls | maximum percentage of missing values allowed |
| Handle duplicate rows | remove them, or replace values in duplicate rows with mean, median, mode or a value |
| Combine columns | concatenate, sum, mean, product, difference |
| Drop columns | remove selected columns |
| Handle outliers | IQR or z-score, then clip or remove |
| Scale | standard, min-max, robust |
| Transform | log1p, square root, square |
| Encode categorical columns | one-hot or label encoding |
| Bin | quantile bins |
| Remove low-variance features | variance threshold |
| Remove correlated features | correlation threshold |
| Dimensionality reduction | PCA, Truncated SVD or t-SNE, with an option to keep columns such as the target untouched |

---

## Explanations

- **What-if (always available):**
  - *Single prediction:* each input is swapped for a typical value (the median, or the most common category) and the change in the prediction is shown. It is an approximation, because overlapping columns can double-count.
  - *Overall:* permutation importance, which shows how much the test score drops when a column's values are shuffled.
- **SHAP (optional):** if `shap` is installed, a method switch appears on the Predict tab. Tree models use SHAP's fast tree method, linear models use its linear method, and KNN, SVM, SVR, Naive Bayes, AdaBoost and Polynomial Regression use the slower general method with capped rows. One-hot columns are summed back to their original column. The baseline is the average over a sample of unseen test rows. The overall chart averages the absolute SHAP values over up to 200 rows (40 for the general method).
- On a small free server, leave SHAP out unless you have tested its memory use. The what-if method covers every model.

---

## Performance and limits

| Area | Behaviour |
|---|---|
| Upload | Up to `MAX_UPLOAD_MB`; the CSV is streamed, and `pyarrow` speeds parsing if installed |
| Datasets in memory | The newest `MAX_DATASETS` are kept. Older ones return a "please reload your dataset" message |
| Dataset IDs | 32-character random IDs |
| Undo history | The last 5 steps plus the original |
| SVM and SVR | Train on a random 10,000-row sample when the training set is larger |
| Cross-validation | Parallel above 5,000 rows, skipped above 20,000 |
| Clustering and anomaly detection | A random sample of 50,000 rows, or 8,000 for Agglomerative, Mean Shift, LOF and One-Class SVM |
| Silhouette score and K scan | Estimated on at most 5,000 points |
| Cluster plots | PCA scatter up to 5,000 points, silhouette analysis up to 400, dendrogram up to 300 rows |
| Tuning | Up to `MAX_TUNE_TRIALS` trials and `MAX_TUNE_SECONDS` seconds per model, on at most `TUNE_MAX_ROWS` rows |
| AutoML | First pass on at most `AUTOML_MAX_ROWS` training rows |
| Background jobs | One per browser tab, and `MAX_CONCURRENT_JOBS` across the server |

The app says when it sampled, through the `trained_rows` column on the Supervised tab and a note above the clustering metrics. Test-set scores always use the full held-out test set.

**Indicative timings** (a 60,000-row synthetic file with 9 columns, on a single-core machine; multi-core machines will be faster for Random Forest, Extra Trees and KNN): upload 0.9 s, Linear Regression plus Random Forest 43 s, SVR on a 10,000-row sample 17 s, KNN 1.6 s, Gradient Boosting 14 s, K-Means with automatic K 10.5 s, Agglomerative on an 8,000-row sample 5.8 s.

**Guidance:** comfortable up to about 50,000 rows, workable to about 200,000 with fewer algorithms at a time, and RAM-limited beyond that.

---

## API reference

Base URL: `http://localhost:8000`. Errors return `{ "detail": "..." }` with a readable message.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/algos` | Algorithm lists and example datasets |
| POST | `/api/sample/{name}` | Load an example dataset |
| POST | `/api/upload` | Upload a CSV (multipart field `file`) |
| GET | `/api/eda/{id}` | Histograms, correlations, box-plot data, statistics |
| GET | `/api/eda/{id}/scatter` | Scatter data (`x`, `y`, optional `color`) |
| GET | `/api/insights/{id}` | Dataset insights |
| POST | `/api/fe/apply` | Apply one feature engineering step |
| POST | `/api/fe/undo/{id}` | Undo the last step |
| POST | `/api/fe/reset/{id}` | Reset to the original data |
| GET | `/api/fe/compare/{id}` | Before vs after data |
| GET | `/api/fe/download/{id}` | Current data as CSV |
| POST | `/api/train` | Train models (blocking) |
| POST | `/api/train/start` | Train models as a background job |
| POST | `/api/cluster` | Cluster or detect anomalies (blocking) |
| POST | `/api/cluster/start` | Cluster as a background job |
| POST | `/api/tune` | Start a tuning job |
| POST | `/api/automl` | Start an AutoML job |
| GET | `/api/job/{id}` | Job status, progress and result |
| POST | `/api/job/{id}/cancel` | Cancel a tuning or AutoML job |
| GET | `/api/models/{dataset_id}` | Models available for a dataset |
| POST | `/api/predict` | Predict for one new row |
| POST | `/api/explain/local` | What-if explanation of one prediction |
| POST | `/api/explain/global` | Permutation importance |
| GET | `/api/explain/capabilities` | Whether SHAP is installed |
| POST | `/api/explain/shap/local` | SHAP explanation of one prediction |
| POST | `/api/explain/shap/global` | Average SHAP importance |
| GET | `/api/model/download` | Download a model (`model_id`) as `.joblib` |

Background job requests send an `X-Client-Id` header (the frontend creates one per browser tab). The server allows one running job per ID and `MAX_CONCURRENT_JOBS` overall, and answers `429` with a message otherwise.

**Train request:**
```json
{
  "dataset_id": "a1b2c3d4...",
  "task": "regression",
  "target": "charges",
  "algos": ["Linear Regression", "Random Forest"],
  "test_size": 0.2,
  "scale": true,
  "drop_duplicates": true,
  "balance": false
}
```

**Cluster request:**
```json
{
  "dataset_id": "a1b2c3d4...",
  "algo": "kmeans",
  "k": 3,
  "k_method": "silhouette",
  "features": ["Age", "Annual Income (k$)"],
  "eps": 1.2,
  "min_samples": 5,
  "contamination": 0.05,
  "label_col": null
}
```
`k_method` can be `manual`, `elbow`, `silhouette`, `calinski`, `davies_bouldin` or `nearest_cluster`.

**Using a downloaded model:**
```python
import joblib, pandas as pd
b = joblib.load("ml-studio-Random_Forest.joblib")   # only load files you trust
pred = b["pipe"].predict(pd.DataFrame([{"column": value}]))[0]
# classification: b["classes"][pred] is the label; b["sklearn_version"] is the version it was saved with
```

---

## Testing and CI

```bash
cd backend
pip install pytest httpx
python -m pytest -q
```

The 27 tests cover dataset detection and insights, EDA data, the full feature engineering flow, training for both task types, balancing and metrics, multiclass ROC, wrong-target rejection, rare classes, prediction and both explanation methods, model download, SHAP (skipped if not installed), clustering with the silhouette plot and dendrogram, the anomaly label check, background jobs and their errors, tuning, AutoML, the job limits and memory eviction.

`.github/workflows/ci.yml` runs the backend tests (Python 3.12) and the frontend build (Node 20) on every push to `main` and on every pull request. Check the **Actions** tab on GitHub for a green tick.

---

## Deployment

### Backend on Render
| Field | Value |
|---|---|
| Root Directory | `backend` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/api/algos` |
| Environment | `PYTHON_VERSION=3.12.7`, `MAX_UPLOAD_MB=25`, `ALLOWED_ORIGINS=https://your-app.vercel.app,http://localhost:5173` |

Do not add `--reload` or extra workers. Datasets and models live in one process's memory, so a second worker would not see them.

### Frontend on Vercel
| Field | Value |
|---|---|
| Root Directory | `frontend` |
| Framework | Vite |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Environment | `VITE_API_URL=https://your-service.onrender.com` |

Deploy the backend first, then set `VITE_API_URL`, then tighten `ALLOWED_ORIGINS`. Every push to `main` redeploys both.

**Free-plan behaviour:** the Render service sleeps after about 15 minutes idle, so the first request can take about a minute. Loaded datasets and trained models are lost whenever it sleeps or restarts. With about 512 MB of RAM, keep uploads small and avoid heavy models on big data.

---

## Design decisions

- **Pipelines prevent leakage:** imputation, encoding and scaling are fitted on the training split only.
- **Separate workflows:** clustering and anomaly detection use internal validation metrics, not a train/test split.
- **One shared dataset state:** feature engineering changes the data every other tab uses, so you can see how cleaning affects model quality.
- **Honest sampling:** algorithms that scale badly on large data use a seeded sample, and the UI says so.
- **Tuning on the training split:** the search sees only cross-validation on training data, so the test score stays an honest estimate.
- **Background jobs with polling:** long runs do not block the server or time out, and they report real progress.
- **Session-only state:** nothing is stored beyond the current visit, so there are no accounts or database to secure.
- **Optional heavy dependencies:** SHAP and pyarrow are used only if installed, so a small server never has to carry them.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Backend not running. Start it on port 8000." | Start `uvicorn main:app --reload --port 8000` in `backend/` and refresh. On the live site, open the Render `/api/algos` URL, wait a minute and refresh |
| "Dataset not found... please reload your dataset" | The server restarted or dropped old data. Load the dataset again |
| "You already have a job running" / "The server is busy" | Wait for the running job or cancel it. Each browser tab gets one job, and the server runs `MAX_CONCURRENT_JOBS` at once |
| Predict tab is empty after changing the data | Feature engineering clears trained models. Train again |
| Yellow notice about the target type | The target column does not match the learning type. Use the one-click switch |
| `cv_mean` and `cv_std` are empty | Cross-validation is skipped above 20,000 training rows, or when a class has only one row |
| `trained_rows` is smaller than the dataset | SVM and SVR train on a 10,000-row sample. Test metrics still use the full test set |
| DBSCAN shows empty metrics | Everything was labelled noise. Increase `eps` |
| Changes not showing | Hard refresh with Ctrl+Shift+R |
| `ModuleNotFoundError` in the backend | Activate the virtual environment and run `pip install -r requirements.txt` |
| A test fails on the models list | An old duplicate `GET /api/models/{dataset_id}` route in `main.py` returns the wrong shape. Keep only the one in `tune.py` |
| CORS error in the browser console | `ALLOWED_ORIGINS` on Render must match your Vercel URL exactly, with `https://` and no trailing slash |
| Requests return 404 on the Vercel domain | `VITE_API_URL` is missing or wrong. Fix it, then redeploy |
| Render build fails on `dotenv` or `optuna` | Make sure `python-dotenv` and `optuna` are in `backend/requirements.txt` |
| SHAP switch is missing | `shap` is not installed on that server. This is expected, and the what-if method still works |

---

## Known limitations

- **No accounts or database.** Datasets, models and history live in memory or in the browser tab and are lost on restart or reload. Export History if you want to keep it.
- **No resampling such as SMOTE.** Balancing uses class weights, which only some models support. KNN, Naive Bayes, Gradient Boosting and AdaBoost ignore it.
- **Time series and text datasets are detected but cannot be trained on.**
- **Large data is sampled** for SVM, SVR and the slower clustering algorithms, so their scores there are approximate.
- **Anomaly check assumption:** the rarer value of the label column is treated as the true outlier.
- **Detection and insights are heuristics** and can be wrong or miss problems.
- **Two example datasets are synthetic.** The real Kaggle fraud data, California Housing and Wholesale Customers are not bundled. Upload them as CSV.
- **Downloaded models are pickles.** Load only files you trust, with the scikit-learn version they were saved with.
- **`SVC(probability=True)`** is deprecated in scikit-learn 1.9 and removed in 1.11, so the dependency is pinned below 1.11.
- **CORS is open by default** for local development. Restrict `ALLOWED_ORIGINS` in production.