# ML Studio: Supervised and Unsupervised Learning Playground

A full-stack machine learning playground. Upload any CSV (or pick an example dataset), explore it, clean and transform it, compare many algorithms side by side, look inside the results with charts, and make new predictions. It has a glassmorphism UI built with React, Tailwind CSS and Bootstrap, and a Python FastAPI backend that runs scikit-learn.

## Live demo

- **App:** https://ml-studio-orpin.vercel.app
- **API docs:** https://ml-studio-e6kr.onrender.com/docs

The backend runs on Render's free tier, so it sleeps when idle. The first request after a quiet period can take about a minute while it wakes up. Loaded datasets and trained models live in memory and are cleared when the service restarts.

---

## Table of contents

1. [Features at a glance](#features-at-a-glance)
2. [Tech stack](#tech-stack)
3. [Project structure](#project-structure)
4. [Setup and run](#setup-and-run)
5. [How to use the app](#how-to-use-the-app)
6. [Tab by tab guide](#tab-by-tab-guide)
7. [Algorithms included](#algorithms-included)
8. [Evaluation metrics](#evaluation-metrics)
9. [Dataset type detection](#dataset-type-detection)
10. [Feature engineering operations](#feature-engineering-operations)
11. [Performance and limits](#performance-and-limits)
12. [API reference](#api-reference)
13. [Design decisions](#design-decisions)
14. [Troubleshooting](#troubleshooting)
15. [Known limitations and what is not built](#known-limitations-and-what-is-not-built)
16. [Roadmap status](#roadmap-status)

---

## Features at a glance

- **Upload any CSV** up to 500 MB, or load a built-in example dataset (grouped as supervised classification, supervised regression and unsupervised clustering).
- **Automatic dataset type detection** (classification, regression, anomaly detection, clustering, time series, text) with a reason, a suggested target column and a confidence level. You can override it.
- **Exploratory data analysis:** correlation heatmap, histograms and summary statistics.
- **Feature engineering studio:** missing values, duplicates, combining columns, outliers and noise, scaling, transforms, encoding, binning, feature selection and dimensionality reduction (PCA, Truncated SVD, t-SNE). Includes a step log, undo (last 5 steps) and reset.
- **Before vs After tab:** shows exactly what your feature engineering changed (shape, missing cells, duplicates, columns, distributions, statistics and correlations).
- **Supervised learning:** 9 regression and 9 classification algorithms, trained and compared in one click, with cross-validation and training time.
- **Rich result visualisations:** metric gauges, confusion matrix, ROC and precision-recall curves, feature importance, actual vs predicted with a perfect-prediction line, residual distribution and a model comparison chart with a metric selector.
- **Unsupervised learning:** 5 clustering algorithms and 3 anomaly detectors, with PCA 2D visualisation and internal validation metrics.
- **Automatic or manual K selection:** elbow, silhouette, Calinski-Harabasz, Davies-Bouldin and nearest cluster distance. Each method's suggestion is shown, and you can override K by hand.
- **Predict tab:** type in new values and get a prediction (with class probabilities for classifiers) from any trained model.
- **Built for larger data:** multi-core training for the heaviest models, automatic sampling for algorithms that scale badly, capped undo history and a parallel-friendly upload path. See [Performance and limits](#performance-and-limits).
- **Reproducible results:** `random_state=42` everywhere a random seed applies, including every sampling step.
- **Polished UX:** glass UI, animated loaders, readable error messages, wrong-target warnings.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Vite 5, Tailwind CSS 3 (preflight disabled), Bootstrap 5.3, Recharts |
| Backend | Python, FastAPI, Uvicorn, Pydantic |
| ML and data | scikit-learn, pandas, NumPy (optional: pyarrow for faster CSV parsing) |
| Styling | Glassmorphism (backdrop blur, translucent cards), gradient background, Sora font (Google Fonts) |
| Communication | REST API (JSON), Vite dev proxy from `/api` to `http://localhost:8000` |

**Prerequisites:** Python 3.10 or newer, Node.js 18 or newer, npm. scikit-learn 1.2 or newer is required (the code uses `OneHotEncoder(sparse_output=False)`).

---

## Project structure

```
ml-studio/
├── README.md
├── backend/
│   ├── main.py             # FastAPI app: datasets, upload, EDA, training, prediction, clustering
│   ├── fe.py               # Feature engineering, undo/reset, before/after comparison, automatic K selection
│   ├── detect.py           # Dataset type detection
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js      # React plugin and /api proxy to port 8000
    ├── postcss.config.js
    ├── tailwind.config.js  # preflight disabled so Tailwind does not fight Bootstrap
    └── src/
        ├── main.jsx        # Entry point (imports Bootstrap CSS, then index.css)
        ├── index.css       # Glass theme, chips, form fixes, loader animations
        ├── App.jsx         # Layout, tabs, Dataset / Explore / Supervised / Unsupervised / Predict
        ├── Engineer.jsx    # Feature engineering tab
        ├── Compare.jsx     # Before vs After tab
        └── Loader.jsx      # Animated loading card
```

---

## Setup and run

### 1. Backend

`requirements.txt`:
```
fastapi
uvicorn[standard]
python-multipart
pandas
numpy
scikit-learn
```

Windows:
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

macOS / Linux:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API is now at `http://localhost:8000`. Interactive API docs are at `http://localhost:8000/docs`.

**Optional: faster CSV parsing for large files**
```bash
pip install pyarrow
```
If `pyarrow` is installed, uploads use it automatically. If it is missing, or does not have builds for your Python version yet, the app falls back to normal pandas parsing with no code changes.

### 2. Frontend

In a second terminal:
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. The Vite dev server forwards every `/api` request to the backend on port 8000, so the backend must be running first.

### 3. Production build (optional)
```bash
cd frontend
npm run build
```

---

## How to use the app

The tabs are laid out in the order of a typical machine learning workflow:

**Dataset → Explore → Engineer → Before vs After → Supervised or Unsupervised → Predict**

1. **Dataset:** upload a CSV or click an example dataset. Read the detected type badge.
2. **Explore:** look at correlations, distributions and statistics.
3. **Engineer:** clean and transform the data step by step. Skip this if your data is already clean.
4. **Before vs After:** check the effect of your steps.
5. **Supervised** (you want to predict a column) or **Unsupervised** (you want to find groups or outliers): pick algorithms, run, and study the charts.
6. **Predict:** enter values and get a prediction from a model you trained.

Suggested order for messy data: remove duplicates, fill missing values, handle outliers, encode text columns, scale, then reduce dimensions if needed.

---

## Tab by tab guide

### Dataset
- **Upload** a `.csv` file (up to 500 MB), or click an example dataset.
- **Examples** are grouped so beginners do not mix them up:

| Group | Dataset | Description |
|---|---|---|
| Supervised: classification | Iris flowers | 150 flowers, predict the species |
| Supervised: classification | Wine types | 178 wines, predict the cultivar |
| Supervised: classification | Breast cancer | 569 tumours, benign or malignant |
| Supervised: regression | Diabetes progression | 442 patients, predict disease progression |
| Unsupervised: clustering | Iris (no labels) | Iris without the species column |
| Unsupervised: clustering | Wine (no labels) | Wine without the cultivar column |
| Unsupervised: clustering | Mall customers | 200 synthetic customers in 5 spending segments |

- **Detected dataset type badge** with the reason, label column, usage hint and confidence. "Not what you expected?" gives two chips: **As detected** and **Treat as unsupervised**. The override changes the guidance and which column starts unselected in the Unsupervised tab. It resets if you apply a feature engineering step.
- **Column table** with type, missing count and unique count, plus the row count, column count and duplicate count.

### Explore
- Correlation heatmap (cyan is positive, pink is negative).
- Histograms for up to 12 numeric columns (15 bins each).
- Summary statistics table (count, mean, std, min, quartiles, max).
- An animated loader shows while the analysis runs.

### Engineer
- Choose a step, set its parameters, select columns (chips show how many nulls each column has) and click **Apply step**.
- **Applied steps** lists every operation with what it changed.
- **Undo last** reverts one step (the last 5 steps can be undone), and **Reset all** returns to the dataset as first loaded.
- The preview table shows the current data. Every other tab uses the current, engineered dataset.
- Applying a step clears previously trained models and cluster results, because the data changed.

See [Feature engineering operations](#feature-engineering-operations) for every operation.

### Before vs After
- **What changed:** rows, columns, missing cells, duplicate rows, numeric and text column counts, each shown as before → after with the difference. Missing cells and duplicates turn green when they drop.
- **Added, removed and type-changed columns.**
- **Steps applied** (the log).
- **Distribution before vs after:** overlaid histograms for any column (up to 15 columns, shared bin range so they are comparable).
- **Column statistics:** mean, standard deviation, skew and missing count, before → after, with changed values in yellow.
- **Correlation change:** heatmap of after minus before for up to 12 numeric columns (cyan = stronger, pink = weaker).
- "Before" is always the dataset as first loaded, kept separately from the undo history, so it is available no matter how many steps you apply. Columns created by feature engineering appear in the Added list but not in the statistics or distribution views, because they have no "before".

### Supervised
- **Setup:** learning type (classification or regression), target column, test size (0.1 to 0.5, default 0.2), scale numeric features, remove duplicates, and the algorithms to compare (pick as many as you like).
- The learning type and target are pre-filled from the dataset detection.
- **Wrong type warnings:** if you choose Classification for a continuous target, or Regression for a text target, a yellow notice offers a one-click switch. The backend also rejects these with a clear message.
- **Pipeline:** each model is wrapped in a scikit-learn Pipeline (median imputation and optional standard scaling for numbers, most-frequent imputation and one-hot encoding for categories). It is fitted on the training split only, so nothing leaks from the test set.
- **Splitting:** `train_test_split` with `random_state=42`, stratified for classification when every class has at least 2 rows.
- **Cross-validation:** 3-fold on the training rows the model used (accuracy for classification, R² for regression).
  - For classification with a rare class, the fold count is lowered to the smallest class size, and skipped if a class has only one row.
  - It is skipped above 20,000 training rows to save time, and runs in parallel above 5,000 rows.
  - When skipped, the `cv_mean` and `cv_std` columns are empty.
- **Large data handling:** SVM and SVR train on a random sample of 10,000 rows when the training set is larger, because they slow down sharply with more rows. The **`trained_rows`** column in the comparison table shows how many rows each model actually learned from. Metrics are always computed on the full held-out test set.
- **Results:**
  - **Model comparison** table (click a row to inspect that model) and a bar chart with a **metric selector**. Classification offers accuracy, precision, recall, F1 and ROC-AUC. Regression offers R², within ±10%, MAE and RMSE. Every model name shows under its bar.
  - **Metric gauges.** Classification: accuracy, precision, recall, F1, ROC-AUC. Regression: R² and "within ±10% of actual", plus MAE and RMSE.
  - **Classification charts:** confusion matrix (heatmap-style), ROC curve and precision-recall curve (binary problems only), feature importance.
  - **Regression charts:** actual vs predicted with a dashed perfect-prediction line and labelled axes, residual distribution, feature importance (where the model provides it).
- An animated loader shows while training.

### Unsupervised
- **Algorithms:** `kmeans`, `dbscan`, `agglomerative`, `gmm`, `meanshift` (clustering), and `isolation_forest`, `lof`, `oneclass_svm` (anomaly detection).
- **Choosing K** (K-Means, Agglomerative and GMM):
  - **Let the algorithm decide K:** pick a method (Elbow, Silhouette, Calinski-Harabasz, Davies-Bouldin or Nearest cluster distance). On Run, K is set automatically.
  - **Assign K manually:** the slider (2 to 10) unlocks. After an automatic run you can click **Change K myself**.
  - After every run you see the K suggested by each method. Click one to use it as your manual K.
- **Features:** numeric columns as chips. The detected label column (if any) starts unselected.
- **Large data handling:** big datasets are randomly sampled before clustering (50,000 rows, or 8,000 for Agglomerative, Mean Shift, LOF and One-Class SVM). The silhouette score is estimated on at most 5,000 points, and the PCA plot draws at most 5,000 points. A note above the metrics tells you when a sample was used and how many rows it contained.
- **Results:**
  - Metrics: silhouette, Davies-Bouldin, Calinski-Harabasz, number of clusters, noise points and K-Means inertia. For anomaly detectors: anomaly and normal counts.
  - **PCA 2D scatter** coloured by cluster, with outliers and noise in red, and the explained variance of each axis in the title.
  - **Elbow** (inertia), **Silhouette by K** and **Nearest cluster distance** curves.
  - **Cluster distribution** bar chart.
- An animated loader shows while the run is in progress.

### Predict
- Pick one of the models you trained on the Supervised tab.
- Numeric features get number inputs (default is the column median). Categorical features get dropdowns.
- Output: the predicted value or class, and class probabilities as progress bars for classifiers that support them.

---

## Algorithms included

### Regression (9)
Linear Regression, Polynomial Regression (degree 2), Ridge, Lasso, Decision Tree, Random Forest, KNN, SVR, Gradient Boosting.

### Classification (9)
Logistic Regression, KNN, Decision Tree, Random Forest, SVM (with probability estimates), Naive Bayes (Gaussian), Gradient Boosting, AdaBoost, Extra Trees.

### Clustering (5)
K-Means, DBSCAN (`eps=1.2`, `min_samples=5` on scaled data), Agglomerative (hierarchical), Gaussian Mixture Model, Mean Shift.

### Anomaly detection (3)
Isolation Forest, Local Outlier Factor, One-Class SVM (contamination or nu fixed at 5%).

### Dimensionality reduction (3)
PCA, Truncated SVD, t-SNE. These are available as a feature engineering step, and PCA also drives the 2D cluster plot.

### Reproducibility
`random_state=42` is set on every model that has a random component, plus the train/test split, PCA, t-SNE, the K-Means scans and every random sampling step. These algorithms have no random element, so they give the same result every run: Linear Regression, KNN, Naive Bayes, DBSCAN, Agglomerative, Mean Shift, LOF and One-Class SVM.

### Multi-core
Random Forest, Extra Trees and KNN use all CPU cores (`n_jobs=-1`).

---

## Evaluation metrics

| Task | Metrics |
|---|---|
| Regression | MAE, MSE, RMSE, R², within ±10% of actual, cross-validated R² (mean and std), training time, rows trained on |
| Classification | Accuracy, precision, recall, F1 (weighted averages), ROC-AUC, confusion matrix, cross-validated accuracy (mean and std), training time, rows trained on. The classification report is computed by the API and returned in the response, but is not shown in the UI yet |
| Clustering | Silhouette score (estimated on at most 5,000 points), Davies-Bouldin index, Calinski-Harabasz score, inertia (K-Means), cluster count, noise count. Noise points (label -1) are excluded from the scores |
| Anomaly detection | Anomaly count, normal count, outliers highlighted on the PCA plot |

**Note on accuracy for regression:** accuracy is a classification metric, so regression has none. The closest equivalents are **R²** (share of variation explained) and **within ±10%** (share of predictions within 10% of the real value).

**Reading actual vs predicted:** each dot is one unseen test row, with the real value across and the model's prediction up. The dashed line is a perfect prediction. Dots above it are overestimates and dots below it are underestimates. The residual chart next to it shows actual minus predicted, and a tall bar centred on 0 with a narrow spread is good.

**How the automatic K methods work:**
- **Elbow:** the knee of the inertia curve, found as the point furthest from the straight line joining the first and last points.
- **Silhouette:** K with the highest silhouette score.
- **Calinski-Harabasz:** K with the highest score.
- **Davies-Bouldin:** K with the lowest score.
- **Nearest cluster distance:** K with the highest ratio of the distance between the two closest cluster centres to the largest average cluster radius (best separated).
- All five are computed from a K-Means scan over K = 2 to 10 on the scaled, median-imputed numeric features (on the sample when the data is large).

---

## Dataset type detection

Detection runs on every upload, example load and feature engineering step, and shows a badge on the Dataset tab.

| Detected type | Rule |
|---|---|
| Anomaly detection (labels available) | A column whose name contains `outlier`, `anomaly`, `anomalous` or `novelty`, with exactly 2 distinct values |
| Time series | A datetime column, or a text column named like `date` or `time` that parses as dates |
| Text (NLP) | A text column with long strings (average length over 50) and mostly unique values |
| Classification (binary or multiclass) | A column named like `target`, `label`, `class`, `outcome`, `species`, `diagnosis`, `survived`, `churn`, `default`, `fraud`, `price`, `salary`, `charges` or `medv` that has few distinct values (text, or whole numbers with at most 20 values and at most 10% of the row count), **or** an unnamed last column with only a few whole-number values (medium confidence) |
| Regression | A named target column (same names as above) that is a continuous number |
| Clustering (no target found) | Anything else |

Detection is a heuristic, so check the confidence label. Time series and text datasets are only flagged. The built-in tabs cannot train on them yet.

---

## Feature engineering operations

| Operation | Options |
|---|---|
| Fill missing values | mean, median, mode, constant, forward fill, KNN imputation (columns optional, empty means all) |
| Drop rows / columns with nulls | rows or columns, with a maximum percentage of missing values allowed |
| Handle duplicate rows | remove them, or replace values in duplicate rows (selected columns) with mean, median, mode or a value you type |
| Combine columns | concatenate (with separator), sum, mean, product, difference, with a name for the new column |
| Drop columns | remove selected columns |
| Handle outliers / noise | IQR or z-score, with a factor (1.5 IQR, 3 for z-score), then clip or remove rows |
| Scale numeric columns | standard, min-max, robust |
| Transform | log1p, square root, square (for skewed or noisy columns) |
| Encode categorical columns | one-hot or label encoding |
| Bin into groups | quantile bins, number of bins you choose |
| Remove low-variance features | variance threshold |
| Remove correlated features | drops one column of each pair above a correlation threshold (default 0.95) |
| Dimensionality reduction | PCA, Truncated SVD or t-SNE (t-SNE limited to 3 components), with a "keep untouched" option for columns such as the target. Output columns are named like `pca_1`, `pca_2` |

Every step is logged with a description of what it did. The original dataset is kept for Reset and Before vs After, and the last 5 steps are kept for Undo.

---

## Performance and limits

### Upload size
- The maximum upload is **500 MB** (`MAX_UPLOAD_MB` in `backend/main.py`). Larger files are rejected with a readable message (HTTP 413).
- Files are parsed straight from disk without an extra copy in memory, and the upload runs in a worker thread so it does not block other requests.
- Parsing is faster if `pyarrow` is installed.
- Rough memory rule: the in-memory table is typically 2 to 5 times the CSV size. The server also keeps the original dataset and up to 5 undo snapshots per dataset, so a very large file used with many feature engineering steps needs several times that.

### What is optimised
| Area | Behaviour |
|---|---|
| Random Forest, Extra Trees, KNN | Use all CPU cores |
| SVM / SVR | Train on a random 10,000-row sample when the training set is larger |
| Cross-validation | Parallel above 5,000 rows, skipped above 20,000 rows |
| Clustering and anomaly detection | Random sample of 50,000 rows, or 8,000 for Agglomerative, Mean Shift, LOF and One-Class SVM |
| Silhouette score and automatic K scan | Estimated on at most 5,000 points |
| PCA cluster plot | Draws at most 5,000 points |
| Undo history | Last 5 steps plus the original |
| Predictions, metrics, curves | Always computed on the full held-out test set (supervised) |

The app tells you when it sampled: the `trained_rows` column on the Supervised tab, and a note above the metrics on the Unsupervised tab.

### Indicative timings
Measured on a synthetic regression file with 60,000 rows and 9 columns, on a **single-core** test machine. On a multi-core computer the Random Forest, Extra Trees and KNN numbers should be lower. Treat these as rough guidance, not guarantees.

| Step | Time |
|---|---|
| Upload | 0.9 s |
| Linear Regression and Random Forest | 43 s |
| SVR (on a 10,000-row sample) | 17 s |
| KNN | 1.6 s |
| Gradient Boosting | 14 s |
| K-Means with automatic K | 10.5 s |
| Agglomerative (on an 8,000-row sample) | 5.8 s |

### Practical guidance (estimates)
| Rows | What to expect |
|---|---|
| Up to about 50,000 | Comfortable for most algorithms |
| 50,000 to 200,000 | Works. Pick fewer algorithms at a time, and expect Random Forest and Gradient Boosting to take the longest |
| Over 200,000 | Limited mostly by your computer's RAM. Sample the data first if you can |

For exact results on large data, prefer the algorithms that are not sampled: Linear, Ridge, Lasso and Logistic Regression, Decision Tree, Random Forest, Naive Bayes, K-Means, GMM and Isolation Forest.

---

## API reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/algos` | Algorithm lists, example datasets (with group, task and description) |
| POST | `/api/sample/{name}` | Load an example dataset (`iris`, `wine`, `breast_cancer`, `diabetes`, `iris_unlabeled`, `wine_unlabeled`, `mall_customers`) |
| POST | `/api/upload` | Upload a CSV (multipart form field `file`, maximum 500 MB, returns 413 above that) |
| GET | `/api/eda/{id}` | Histograms, correlation matrix, summary statistics |
| POST | `/api/fe/apply` | Apply one feature engineering step |
| POST | `/api/fe/undo/{id}` | Undo the last step (up to 5 steps back) |
| POST | `/api/fe/reset/{id}` | Return to the dataset as first loaded |
| GET | `/api/fe/compare/{id}` | Before vs after comparison data |
| POST | `/api/train` | Train and evaluate one or more supervised models |
| POST | `/api/predict` | Predict for one new row with a trained model |
| POST | `/api/cluster` | Run clustering or anomaly detection |

Dataset responses include `id`, `rows`, `cols` (name, dtype, missing, unique, numeric), `duplicates`, `preview`, `detected`, and (after feature engineering) `log`.

**Train request:**
```json
{
  "dataset_id": "a1b2c3d4",
  "task": "regression",
  "target": "charges",
  "algos": ["Linear Regression", "Random Forest"],
  "test_size": 0.2,
  "scale": true,
  "drop_duplicates": true
}
```
The response lists, for each model, its metrics, `cv_mean`, `cv_std`, `trained_rows` and `train_time`, plus the chart data for each model.

**Cluster request:**
```json
{
  "dataset_id": "a1b2c3d4",
  "algo": "kmeans",
  "k": 3,
  "features": ["Age", "Annual Income (k$)"],
  "k_method": "silhouette"
}
```
Set `k_method` to `manual` to use `k` as given. Other values are `elbow`, `silhouette`, `calinski`, `davies_bouldin` and `nearest_cluster`. The response includes `k_used`, `k_picks`, `metrics`, `points` (at most 5,000), `elbow` (the K scan), `counts`, `var`, and `sampled_rows` and `total_rows`.

**Feature engineering request:**
```json
{
  "dataset_id": "a1b2c3d4",
  "op": "fill_missing",
  "cols": ["bmi"],
  "params": { "strategy": "median" }
}
```
Operation names: `fill_missing`, `drop_missing`, `duplicates`, `combine`, `drop_columns`, `outliers`, `scale`, `transform`, `encode`, `bin`, `var_threshold`, `corr_filter`, `reduce`.

**Predict request:** `{ "model_id": "<dataset_id>:<model name>", "row": { "age": 30, "smoker": "yes" } }`

Errors come back as JSON `{ "detail": "..." }` with a readable message (for example, choosing Classification for a continuous column, or uploading a file that is too large).

---

## Design decisions

- **Pipelines prevent data leakage:** imputation, encoding and scaling are fitted on the training split only.
- **Separate workflows for supervised and unsupervised learning:** clustering and anomaly detection use internal validation metrics rather than a train/test split.
- **One shared dataset state:** feature engineering changes the dataset used by every other tab, so the effect of cleaning on model quality is easy to see.
- **Honest sampling:** algorithms that scale badly on large data use a seeded random sample, and the UI says so instead of silently changing results.
- **Glass UI with Bootstrap and Tailwind together:** Bootstrap provides the grid, forms and tables. Tailwind provides utility classes. Tailwind's preflight reset is turned off so the two do not conflict.
- **In-memory storage:** datasets, history and trained models live in server memory for speed and simplicity. The history is capped so memory use stays predictable.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Backend not running. Start it on port 8000." | Start `uvicorn main:app --reload --port 8000` in `backend/` and refresh |
| Blank page or red error after pasting code | Open the browser console (F12), read the first red error, and check the file it names |
| `list is not defined` on the Supervised tab | In `function Train`, make sure the line `const list = meta[task]` is still there |
| "Unexpected token ... is not valid JSON" | You are on an old version of the API helper. Update it so non-JSON errors show a readable message |
| Changes not showing | Hard refresh with Ctrl+Shift+R |
| `ModuleNotFoundError` in the backend | Activate the virtual environment and run `pip install -r requirements.txt` again |
| `sparse_output` error | Upgrade scikit-learn: `pip install -U scikit-learn` |
| Port already in use | Stop the other process, or change the port in the `uvicorn` command and in `vite.config.js` |
| "Choose Regression instead" / "Choose Classification instead" | The target column does not match the learning type. Use the one-click switch in the yellow notice |
| "File is larger than 500 MB" | Upload a smaller sample, or raise `MAX_UPLOAD_MB` in `backend/main.py` if your computer has enough RAM |
| Upload or training is slow on a big file | Pick fewer algorithms. Random Forest, Gradient Boosting and SVM are the slowest. Installing `pyarrow` speeds up file parsing |
| `cv_mean` and `cv_std` are empty | Cross-validation is skipped above 20,000 training rows, or when a class has only one row |
| `trained_rows` is smaller than your dataset | SVM and SVR train on a 10,000-row sample for speed. Test metrics still use the full test set |
| Unsupervised note says a sample was used | The data was larger than the sampling cap for that algorithm. See [Performance and limits](#performance-and-limits) |
| Undo does nothing | Only the last 5 steps can be undone. Use **Reset all** to go back to the original |
| Predictions fail after changing data | Retrain: feature engineering clears trained models, and the Predict tab needs a fresh one |

---

## Known limitations and what is not built

**Data and storage**
- Datasets, history and trained models are stored in memory and are lost when the backend restarts. There are no user accounts, saved experiments or database (MongoDB) in this version, and models are not saved to disk (no `joblib`).
- CORS is open for local development. Do not expose the backend publicly as is.

**Scale**
- Very large files are limited by RAM. SVM, SVR and the slower clustering algorithms use samples on big data, so their scores there are approximate.
- Training and clustering run as a single request, so the loader shows an elapsed-seconds counter but not real per-model progress. The status messages rotate on a timer.

**Visualisations not included yet**
- Box plots and scatter plots in Explore, a dendrogram for hierarchical clustering, a regression line plot, a per-sample silhouette plot, a standalone t-SNE/PCA visualisation page, and a precision-recall or ROC curve for multiclass problems (only the ROC-AUC number is shown).
- The classification report is calculated but not displayed.

**Learning features not included yet**
- Hyperparameter tuning (Optuna or grid search), AutoML, SHAP explanations, imbalanced-data handling (`imbalanced-learn`), a "Learning mode" that explains each algorithm, experiment history and comparison, and automatic dataset insights.
- Time series and text datasets are detected but cannot be trained on.
- DBSCAN `eps` and the anomaly contamination rate (5%) are fixed. They are not adjustable in the UI.

**Datasets not bundled**
- California Housing, Wholesale Customers and Credit Card Fraud are not built in. Download them (UCI, Kaggle or scikit-learn) and upload as CSV.

**Heuristics**
- Dataset type detection relies on column names and value patterns. It can be wrong, which is why the override chips exist.

---

## Roadmap status

| Phase | Work | Status |
|---|---|---|
| 1 | Dataset upload and EDA | Done |
| 2 | Data preprocessing and feature engineering | Done (extended well beyond the original plan) |
| 3 | Linear and Logistic Regression | Done |
| 4 | KNN, Decision Tree, Random Forest, SVM and more | Done (18 supervised algorithms) |
| 5 | K-Means, DBSCAN, Hierarchical | Done (plus GMM, Mean Shift) |
| 6 | Train/test split and evaluation metrics | Done |
| 7 | Interactive graphs and model comparison | Done |
| 8 | PCA, anomaly detection, dimensionality reduction | Done (as an engineering step and in the cluster view) |
| 9 | AutoML and hyperparameter tuning | Not started |
| 10 | SHAP explanations and experiment history | Not started |
| Extra | Large-file support and speed (500 MB uploads, multi-core, sampling) | Done |

Ideas for next steps: hyperparameter tuning, a classification report view, box and scatter plots, a dendrogram, saved experiments with MongoDB, model download with `joblib`, SHAP explanations, real per-model progress (streaming), and checking anomaly detectors against a labelled outlier column.