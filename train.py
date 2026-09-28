"""Run the full offline pipeline: data -> cleaning -> KPIs -> scoring -> models.

Usage:
    python train.py                 # reuse data/raw/vendor_orders.csv if it exists
    python train.py --regenerate    # rebuild the synthetic raw dataset first
"""
import argparse
import json

import joblib
import pandas as pd

from src import config
from src.cleaning import run_cleaning
from src.data_generation import build_raw_dataset
from src.features import add_historical_features, vendor_snapshot
from src.kpi import compute_vendor_kpis
from src.models import fit_clusters, train_classifier
from src.scoring import score_vendors


def main(regenerate: bool) -> None:
    config.MODELS_DIR.mkdir(exist_ok=True)
    if regenerate or not config.RAW_CSV.exists():
        print("Generating synthetic raw data ...")
        build_raw_dataset()

    print("Cleaning ...")
    clean = run_cleaning()

    print("KPIs and scoring ...")
    scored = score_vendors(compute_vendor_kpis(clean))

    print("Clustering ...")
    clusters = fit_clusters(scored)
    clusters["vendors"].to_csv(config.KPI_CSV, index=False)
    clusters["summary"].to_csv(config.CLUSTER_SUMMARY_CSV, index=False)
    clusters["selection"].to_csv(config.KMEANS_SELECTION_CSV, index=False)
    joblib.dump(clusters["artefact"], config.CLUSTER_MODEL_PKL)
    print(f"  chosen k = {clusters['k']}")
    print(clusters["summary"].to_string(index=False))

    print("Late-delivery classifier ...")
    model_df = add_historical_features(clean)
    model_df.to_csv(config.MODEL_DATA_CSV, index=False)
    vendor_snapshot(model_df).to_csv(config.VENDOR_SNAPSHOT_CSV, index=False)
    result = train_classifier(model_df)
    joblib.dump(result["model"], config.CLASSIFIER_PKL)
    result["impurity_importance"].to_csv(config.FEATURE_IMPORTANCE_CSV, index=False)
    result["permutation_importance"].to_csv(config.PERMUTATION_IMPORTANCE_CSV, index=False)

    metrics = {
        "best_model": result["best_model"], "split": result["split"],
        "comparison": result["comparison"].to_dict("records"),
        "final": result["final_metrics"], "kmeans_k": clusters["k"],
        "decision_threshold": config.DECISION_THRESHOLD,
    }
    config.METRICS_JSON.write_text(json.dumps(metrics, indent=2, default=float))

    pd.set_option("display.width", 200)
    print(result["comparison"].drop(columns="confusion_matrix").round(3).to_string(index=False))
    print("Best model:", result["best_model"])
    print(json.dumps(result["final_metrics"], indent=2, default=float))
    print(result["permutation_importance"].head(8).round(4).to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true", help="rebuild the raw synthetic dataset")
    main(parser.parse_args().regenerate)
