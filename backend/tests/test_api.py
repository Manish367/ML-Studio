import io, time
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import main
from main import app


@pytest.fixture(scope="module")
def c():
    return TestClient(app)


def upload(c, df):
    r = c.post(
        "/api/upload", files={"file": ("d.csv", df.to_csv(index=False), "text/csv")}
    )
    assert r.status_code == 200, r.text
    return r.json()


def sample(c, name):
    r = c.post(f"/api/sample/{name}")
    assert r.status_code == 200, r.text
    return r.json()


def wait(c, job_id, limit=120):
    t = time.time()
    while time.time() - t < limit:
        j = c.get(f"/api/job/{job_id}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish in time")


def synthetic(n=300, seed=0):
    rng = np.random.RandomState(seed)
    d = pd.DataFrame(
        {"x": rng.randn(n), "z": rng.randn(n), "grp": rng.choice(["a", "b", "c"], n)}
    )
    d["label"] = ((d.x > 0) ^ (d.grp == "a")).astype(int)
    d["amount"] = d.x * 5 + (d.grp == "b") * 3 + rng.randn(n) * 0.3
    return d


# ---------- datasets and detection ----------
def test_algos_and_samples(c):
    r = c.get("/api/algos").json()
    assert len(r["regression"]) == 9 and len(r["classification"]) == 9
    ids = {s["id"] for s in r["samples"]}
    assert {"iris", "diabetes", "mall_customers", "fraud_demo", "wholesale_demo"} <= ids


@pytest.mark.parametrize(
    "name,task",
    [
        ("iris", "classification"),
        ("diabetes", "regression"),
        ("mall_customers", "clustering"),
        ("fraud_demo", "classification"),
    ],
)
def test_detection_on_samples(c, name, task):
    assert sample(c, name)["detected"]["task"] == task


def test_detects_labelled_anomaly_dataset(c):
    rng = np.random.RandomState(1)
    d = pd.DataFrame(
        {
            "a": rng.randn(500),
            "b": rng.randn(500),
            "Outlier_label": (rng.rand(500) < 0.03).astype(int),
        }
    )
    assert upload(c, d)["detected"]["task"] == "anomaly"


def test_insights_flag_problems(c):
    d = synthetic()
    d["const"] = 1
    d.loc[:200, "z"] = np.nan
    d = pd.concat([d, d.head(30)])
    titles = [
        x["title"] for x in c.get(f"/api/insights/{upload(c, d)['id']}").json()["items"]
    ]
    assert any("duplicate" in t for t in titles)
    assert any("single value" in t for t in titles)
    assert any("mostly empty" in t for t in titles)


def test_eda_box_and_scatter(c):
    i = upload(c, synthetic())["id"]
    eda = c.get(f"/api/eda/{i}").json()
    assert {"min", "q1", "median", "q3", "max", "outliers"} <= set(eda["box"]["x"])
    pts = c.get(
        f"/api/eda/{i}/scatter", params={"x": "x", "y": "z", "color": "grp"}
    ).json()["points"]
    assert {p["c"] for p in pts} == {"a", "b", "c"}
    assert (
        c.get(f"/api/eda/{i}/scatter", params={"x": "x", "y": "grp"}).status_code == 400
    )


def test_missing_dataset_gives_friendly_404(c):
    r = c.get("/api/eda/does-not-exist")
    assert r.status_code == 404 and "reload" in r.json()["detail"].lower()


# ---------- feature engineering ----------
def test_feature_engineering_flow(c):
    d = synthetic()
    d.loc[:20, "x"] = np.nan
    i = upload(c, d)["id"]
    r = c.post(
        "/api/fe/apply",
        json={
            "dataset_id": i,
            "op": "fill_missing",
            "cols": ["x"],
            "params": {"strategy": "median"},
        },
    ).json()
    assert (
        next(col for col in r["cols"] if col["name"] == "x")["missing"] == 0
        and len(r["log"]) == 1
    )
    c.post(
        "/api/fe/apply",
        json={
            "dataset_id": i,
            "op": "combine",
            "cols": ["x", "z"],
            "params": {"how": "sum", "name": "xz"},
        },
    )
    cmp = c.get(f"/api/fe/compare/{i}").json()
    assert cmp["before"]["missing"] > 0 and cmp["after"]["missing"] == 0
    assert [a["name"] for a in cmp["added"]] == ["xz"]
    csv = c.get(f"/api/fe/download/{i}")
    assert csv.status_code == 200 and csv.text.splitlines()[0].startswith("x,z")
    assert len(c.post(f"/api/fe/undo/{i}").json()["log"]) == 1
    assert c.post(f"/api/fe/reset/{i}").json()["log"] == []


def test_feature_engineering_clears_stale_models(c):
    i = upload(c, synthetic())["id"]
    c.post(
        "/api/train",
        json={
            "dataset_id": i,
            "task": "classification",
            "target": "label",
            "algos": ["KNN"],
        },
    )
    assert len(c.get(f"/api/models/{i}").json()) == 1
    c.post("/api/fe/apply", json={"dataset_id": i, "op": "scale", "params": {}})
    assert c.get(f"/api/models/{i}").json() == []


# ---------- supervised learning ----------
def test_train_classification_and_metrics(c):
    i = upload(c, synthetic())["id"]
    r = c.post(
        "/api/train",
        json={
            "dataset_id": i,
            "task": "classification",
            "target": "label",
            "algos": ["Random Forest", "KNN"],
            "balance": True,
        },
    )
    assert r.status_code == 200
    res = r.json()
    assert {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    } <= set(res["results"][0])
    d = res["details"]["Random Forest"]
    assert d["confusion"] and d["roc"] and d["report"] and d["importance"]


def test_train_regression_and_metrics(c):
    i = upload(c, synthetic())["id"]
    res = c.post(
        "/api/train",
        json={
            "dataset_id": i,
            "task": "regression",
            "target": "amount",
            "algos": ["Ridge", "Random Forest"],
        },
    ).json()
    assert {"mae", "rmse", "r2", "within_10pct"} <= set(res["results"][0])
    assert res["details"]["Ridge"]["points"] and res["details"]["Ridge"]["residuals"]


def test_multiclass_roc_curves(c):
    i = sample(c, "wine")["id"]
    d = c.post(
        "/api/train",
        json={
            "dataset_id": i,
            "task": "classification",
            "target": "target",
            "algos": ["Random Forest"],
        },
    ).json()["details"]["Random Forest"]
    assert len(d["roc_multi"]) == 3


def test_wrong_learning_type_is_rejected(c):
    i = upload(c, synthetic())["id"]
    r = c.post(
        "/api/train",
        json={
            "dataset_id": i,
            "task": "classification",
            "target": "amount",
            "algos": ["KNN"],
        },
    )
    assert r.status_code == 400 and "Regression" in r.json()["detail"]
    r = c.post(
        "/api/train",
        json={
            "dataset_id": i,
            "task": "regression",
            "target": "grp",
            "algos": ["Ridge"],
        },
    )
    assert r.status_code == 400 and "Classification" in r.json()["detail"]


def test_rare_class_does_not_crash(c):
    d = synthetic()
    d.loc[0, "grp"] = "rare"
    i = upload(c, d)["id"]
    assert (
        c.post(
            "/api/train",
            json={
                "dataset_id": i,
                "task": "classification",
                "target": "grp",
                "algos": ["Random Forest"],
            },
        ).status_code
        == 200
    )


def test_predict_and_explanations(c):
    i = upload(c, synthetic())["id"]
    c.post(
        "/api/train",
        json={
            "dataset_id": i,
            "task": "classification",
            "target": "label",
            "algos": ["Random Forest"],
        },
    )
    body = {
        "model_id": f"{i}:Random Forest",
        "row": {"x": 1.5, "z": 0.0, "amount": 7.0, "grp": "a"},
    }
    p = c.post("/api/predict", json=body).json()
    assert p["prediction"] in ("0", "1") and abs(sum(p["proba"].values()) - 1) < 1e-6
    loc = c.post("/api/explain/local", json=body).json()
    assert loc["items"] and "baseline" in loc
    glob = c.post("/api/explain/global", json={"model_id": body["model_id"]}).json()
    assert glob["items"][0]["feature"] in ("x", "grp", "amount")
    assert (
        c.post("/api/predict", json={"model_id": "nope:Model", "row": {}}).status_code
        == 404
    )


def test_model_download_roundtrip(c):
    i = upload(c, synthetic())["id"]
    c.post(
        "/api/train",
        json={
            "dataset_id": i,
            "task": "classification",
            "target": "label",
            "algos": ["Random Forest"],
        },
    )
    r = c.get("/api/model/download", params={"model_id": f"{i}:Random Forest"})
    assert r.status_code == 200 and "joblib" in r.headers["content-disposition"]
    bundle = joblib.load(io.BytesIO(r.content))
    pred = bundle["pipe"].predict(
        pd.DataFrame([{"x": 1.5, "z": 0.0, "amount": 7.0, "grp": "a"}])
    )[0]
    assert bundle["classes"][pred] in ("0", "1")


def test_shap_explanations_when_installed(c):
    pytest.importorskip("shap")
    assert c.get("/api/explain/capabilities").json()["shap"] is True
    i = upload(c, synthetic())["id"]
    c.post(
        "/api/train",
        json={
            "dataset_id": i,
            "task": "regression",
            "target": "amount",
            "algos": ["Random Forest", "Ridge"],
        },
    )
    row = {"x": 1.5, "z": 0.0, "grp": "b", "label": 1}
    for model in ("Random Forest", "Ridge"):
        body = {"model_id": f"{i}:{model}", "row": row}
        loc = c.post("/api/explain/shap/local", json=body).json()
        assert (
            abs(
                loc["baseline"]
                + sum(x["contribution"] for x in loc["items"])
                - loc["full"]
            )
            < 0.5
        )
        assert c.post(
            "/api/explain/shap/global", json={"model_id": body["model_id"]}
        ).json()["items"]


# ---------- unsupervised learning ----------
def test_cluster_automatic_k_and_plots(c):
    i = sample(c, "mall_customers")["id"]
    feats = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    r = c.post(
        "/api/cluster",
        json={
            "dataset_id": i,
            "algo": "agglomerative",
            "k_method": "silhouette",
            "features": feats,
        },
    ).json()
    assert 2 <= r["k_used"] <= 10 and set(r["k_picks"]) == {
        "elbow",
        "silhouette",
        "calinski",
        "davies_bouldin",
        "nearest_cluster",
    }
    assert (
        r["silhouette_plot"]["clusters"]
        and r["dendro"]["i"]
        and len(r["points"]) <= 5000
    )
    m = c.post(
        "/api/cluster",
        json={
            "dataset_id": i,
            "algo": "kmeans",
            "k": 4,
            "k_method": "manual",
            "features": feats,
        },
    ).json()
    assert m["k_used"] == 4 and "inertia" in m["metrics"]


def test_anomaly_detection_against_labels(c):
    rng = np.random.RandomState(0)
    X = rng.randn(800, 3)
    lab = np.zeros(800, int)
    out = rng.choice(800, 32, replace=False)
    X[out] += rng.choice([-1, 1], (32, 3)) * 6
    lab[out] = 1
    i = upload(
        c,
        pd.DataFrame(
            {"f1": X[:, 0], "f2": X[:, 1], "f3": X[:, 2], "Outlier_label": lab}
        ),
    )["id"]
    r = c.post(
        "/api/cluster",
        json={
            "dataset_id": i,
            "algo": "isolation_forest",
            "features": ["f1", "f2", "f3"],
            "contamination": 0.04,
            "label_col": "Outlier_label",
        },
    ).json()
    assert r["label_check"]["recall"] > 0.8 and r["metrics"]["anomalies"] > 0


# ---------- background jobs ----------
def test_training_job_reports_progress(c):
    i = sample(c, "wine")["id"]
    r = c.post(
        "/api/train/start",
        json={
            "dataset_id": i,
            "task": "classification",
            "target": "target",
            "algos": ["Random Forest", "KNN"],
        },
        headers={"X-Client-Id": "t1"},
    )
    assert r.status_code == 200
    j = wait(c, r.json()["job_id"])
    assert (
        j["status"] == "done" and j["pct"] == 100 and len(j["result"]["results"]) == 2
    )


def test_job_errors_are_readable(c):
    i = sample(c, "wine")["id"]
    j = wait(
        c,
        c.post(
            "/api/train/start",
            json={
                "dataset_id": i,
                "task": "classification",
                "target": "proline",
                "algos": ["KNN"],
            },
            headers={"X-Client-Id": "t2"},
        ).json()["job_id"],
    )
    assert j["status"] == "error" and "continuous" in j["error"]


def test_clustering_job(c):
    i = sample(c, "iris_unlabeled")["id"]
    j = wait(
        c,
        c.post(
            "/api/cluster/start",
            json={"dataset_id": i, "algo": "kmeans", "k_method": "elbow"},
            headers={"X-Client-Id": "t3"},
        ).json()["job_id"],
    )
    assert j["status"] == "done" and j["result"]["k_used"] >= 2


def test_tuning_and_automl_jobs(c):
    i = upload(c, synthetic())["id"]
    j = wait(
        c,
        c.post(
            "/api/tune",
            json={
                "dataset_id": i,
                "task": "classification",
                "target": "label",
                "algos": ["KNN", "Naive Bayes"],
                "trials": 5,
                "seconds": 5,
            },
            headers={"X-Client-Id": "t4"},
        ).json()["job_id"],
    )
    assert j["status"] == "done" and all("baseline" in m for m in j["result"]["models"])
    j = wait(
        c,
        c.post(
            "/api/automl",
            json={
                "dataset_id": i,
                "task": "classification",
                "target": "label",
                "trials": 5,
                "seconds": 5,
                "balance": True,
            },
            headers={"X-Client-Id": "t5"},
        ).json()["job_id"],
    )
    assert j["status"] == "done" and j["result"]["winner"] and j["result"]["summary"]
    names = [m["name"] for m in c.get(f"/api/models/{i}").json()]
    assert any(n.startswith("AutoML best") for n in names) and any(
        n.endswith("(tuned)") for n in names
    )


def test_one_job_per_visitor_and_server_limit(c):
    i = upload(c, synthetic(800))["id"]
    body = {
        "dataset_id": i,
        "task": "classification",
        "target": "label",
        "algos": ["Random Forest"],
        "trials": 30,
        "seconds": 30,
    }
    a = c.post("/api/tune", json=body, headers={"X-Client-Id": "A"})
    assert a.status_code == 200
    assert (
        c.post("/api/tune", json=body, headers={"X-Client-Id": "A"}).status_code == 429
    )
    b = c.post("/api/tune", json=body, headers={"X-Client-Id": "B"})
    assert b.status_code == 200
    assert (
        c.post("/api/tune", json=body, headers={"X-Client-Id": "C"}).status_code == 429
    )
    for r in (a, b):
        c.post(f"/api/job/{r.json()['job_id']}/cancel")
        wait(c, r.json()["job_id"])


# ---------- memory limits (keep this last: it loads many datasets) ----------
def test_old_datasets_are_evicted(c):
    ids = [
        upload(c, pd.DataFrame({"a": range(20), "b": range(20)}))["id"]
        for _ in range(main.MAX_DATASETS + 2)
    ]
    assert len(ids[0]) == 32 and ids[0] not in main.DATA and ids[-1] in main.DATA
    assert len(main.DATA) <= main.MAX_DATASETS
