"""
Drift monitoring with Evidently AI.

Two monitors are implemented:
  1. DataDriftMonitor   — compares production input feature distributions
                          against the training reference set.
  2. TargetDriftMonitor — detects concept drift by comparing model
                          prediction distributions over time.

Usage (standalone):
    python -m src.monitoring.drift_detection \
        --reference data/creditcard_fraud.csv \
        --current   data/production_sample.csv \
        --report-dir reports/

In CI / scheduled jobs this module is imported and called directly.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Lazy imports so the rest of the app doesn't hard-depend on Evidently
# ------------------------------------------------------------------
def _import_evidently():
    try:
        # evidently 0.7+ moved the original Report/Preset API to evidently.legacy.*
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
        from evidently.legacy.metrics import DatasetDriftMetric
        return Report, DataDriftPreset, TargetDriftPreset, DataQualityPreset, DatasetDriftMetric
    except ImportError as e:
        raise ImportError(
            "evidently is not installed. Run: pip install evidently"
        ) from e


def _import_evidently_workspace():
    """Import the evidently 0.7.x workspace / UI API (optional dependency)."""
    try:
        from evidently.ui.workspace import Workspace
        return Workspace
    except ImportError as e:
        raise ImportError(
            "evidently workspace UI requires 'evidently[ui]'. "
            "Run: pip install 'evidently[ui]'"
        ) from e


# ------------------------------------------------------------------
# Feature columns (PCA features + engineered)
# ------------------------------------------------------------------
V_FEATURES = [f"V{i}" for i in range(1, 29)]
ENGINEERED_FEATURES = ["log_amount", "hour"]
ALL_FEATURES = V_FEATURES + ENGINEERED_FEATURES
TARGET_COL = "Class"


# ------------------------------------------------------------------
# Data prep helpers
# ------------------------------------------------------------------

def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same transformations as the feature pipeline."""
    df = df.copy()
    if "Amount" in df.columns:
        df["log_amount"] = np.log1p(df["Amount"])
    if "Time" in df.columns:
        df["hour"] = (df["Time"] % 86400) / 3600
    df.drop(columns=[c for c in ["Amount", "Time"] if c in df.columns], inplace=True)
    return df


def _load_and_prepare(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows)
    return _engineer(df)


# ------------------------------------------------------------------
# Monitor classes
# ------------------------------------------------------------------

class DataDriftMonitor:
    """
    Compares feature distributions between a reference (training) dataset
    and a current (production) dataset using Evidently's DataDriftPreset.

    Reports are saved as:
      - Standalone HTML  (open in any browser, no server needed)
      - JSON             (raw data for programmatic consumption)
      - Evidently workspace (optional — enables ``evidently ui`` dashboard)

    Usage (local UI dashboard)
    --------------------------
    monitor = DataDriftMonitor(..., workspace_dir="evidently_workspace")
    monitor.run(current_df)
    # then: evidently ui --workspace evidently_workspace --port 8080
    """

    def __init__(
        self,
        reference_path: str,
        report_dir: str = "reports",
        reference_nrows: Optional[int] = None,
        workspace_dir: Optional[str] = None,   # set to enable evidently ui
        project_name: str = "Fraud Detection Monitoring",
    ):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir = workspace_dir
        self.project_name = project_name

        logger.info("Loading reference data from %s …", reference_path)
        ref_df = _load_and_prepare(reference_path, nrows=reference_nrows)
        # Keep only feature columns for drift analysis (exclude target)
        self.reference = ref_df[[c for c in ALL_FEATURES if c in ref_df.columns]]
        logger.info("Reference set: %d rows × %d features", *self.reference.shape)

    def run(
        self,
        current: pd.DataFrame,
        report_name: Optional[str] = None,
    ) -> dict:
        """
        Run data drift analysis against ``current`` DataFrame.

        Returns a summary dict:
          - dataset_drift (bool): whether overall drift was detected
          - drifted_features (list[str]): features with drift
          - drift_share (float): share of features drifted
          - report_path (str): path to the saved HTML report
        """
        Report, DataDriftPreset, _, _, DatasetDriftMetric = _import_evidently()

        # Prepare current — engineer + align columns
        if "Amount" in current.columns or "Time" in current.columns:
            current = _engineer(current)
        curr_features = current[[c for c in ALL_FEATURES if c in current.columns]]

        report = Report(metrics=[
            DataDriftPreset(),
            DatasetDriftMetric(),
        ])
        report.run(reference_data=self.reference, current_data=curr_features)

        # Save standalone HTML + JSON (always)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        name = report_name or f"data_drift_{ts}"
        html_path = self.report_dir / f"{name}.html"
        json_path = self.report_dir / f"{name}.json"
        report.save_html(str(html_path))

        result = report.as_dict()
        json_path.write_text(json.dumps(result, default=str))

        # Optionally save into an Evidently workspace (enables evidently ui)
        if self.workspace_dir:
            self._save_to_workspace(report)

        summary = _parse_drift_summary(result)
        summary["report_path"] = str(html_path)
        summary["json_path"] = str(json_path)

        logger.info(
            "Data drift: dataset_drift=%s  drifted=%d/%d features  share=%.2f",
            summary["dataset_drift"],
            len(summary["drifted_features"]),
            len(ALL_FEATURES),
            summary["drift_share"],
        )
        return summary

    def _save_to_workspace(self, report) -> None:
        """Save report into an Evidently UI workspace for the local dashboard."""
        try:
            Workspace = _import_evidently_workspace()
            ws = Workspace.create(self.workspace_dir)
            # Re-use an existing project by name, or create one
            projects = {p.name: p for p in ws.list_projects()}
            if self.project_name in projects:
                project = projects[self.project_name]
            else:
                project = ws.create_project(self.project_name)
                project.description = "Fraud detection feature drift monitoring"
                project.save()
            ws.add_report(project.id, report)
            logger.info(
                "Report saved to workspace %s (project: %s). "
                "Run: evidently ui --workspace %s --port 8080",
                self.workspace_dir, self.project_name, self.workspace_dir,
            )
        except Exception as e:
            logger.warning("Could not save to Evidently workspace: %s", e)


class TargetDriftMonitor:
    """
    Detects concept drift by comparing model prediction distributions
    between a reference window and the current window.
    """

    def __init__(
        self,
        reference_predictions: pd.Series,
        report_dir: str = "reports",
    ):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.reference_preds = reference_predictions.rename("prediction")

    def run(
        self,
        current_predictions: pd.Series,
        report_name: Optional[str] = None,
    ) -> dict:
        """Compare prediction distributions; return summary dict."""
        _, _, TargetDriftPreset, _, _ = _import_evidently()
        from evidently.legacy.report import Report

        ref_df = pd.DataFrame({"prediction": self.reference_preds})
        cur_df = pd.DataFrame({"prediction": current_predictions.rename("prediction")})

        report = Report(metrics=[TargetDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        name = report_name or f"target_drift_{ts}"
        html_path = self.report_dir / f"{name}.html"
        report.save_html(str(html_path))

        result = report.as_dict()
        summary = {"report_path": str(html_path)}

        # Try to extract drift flag from result metrics
        try:
            for metric in result.get("metrics", []):
                if "DatasetDriftMetric" in metric.get("metric", ""):
                    summary["dataset_drift"] = metric["result"].get("dataset_drift", False)
        except Exception:
            pass

        logger.info("Target drift report saved to %s", html_path)
        return summary


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_drift_summary(result: dict) -> dict:
    """Extract drift summary from Evidently report dict.

    Handles two evidently result structures:
      - evidently < 0.7: one ColumnDriftMetric entry per column, each with
        ``column_name`` and ``drift_detected`` at the top level of ``result``.
      - evidently >= 0.7 (legacy API): one DataDriftTable entry with all
        per-column results nested inside ``result["drift_by_columns"]``.
    """
    drifted_features = []
    drift_share = 0.0
    dataset_drift = False

    try:
        for metric in result.get("metrics", []):
            mtype = metric.get("metric", "")
            r = metric.get("result", {})

            if "DataDriftTable" in mtype:
                # evidently 0.7.x: per-column results are in drift_by_columns
                for col_name, col_result in r.get("drift_by_columns", {}).items():
                    if col_result.get("drift_detected", False):
                        drifted_features.append(col_name)

            elif "ColumnDriftMetric" in mtype:
                # evidently < 0.7: flat per-column metric entry
                if r.get("drift_detected", False):
                    col = r.get("column_name", "unknown")
                    drifted_features.append(col)

            if "DatasetDriftMetric" in mtype:
                dataset_drift = r.get("dataset_drift", False)
                drift_share = r.get("share_of_drifted_columns", 0.0)

    except Exception as e:
        logger.warning("Error parsing drift result: %s", e)

    return {
        "dataset_drift": dataset_drift,
        "drifted_features": drifted_features,
        "drift_share": drift_share,
    }


def generate_monitoring_report(
    reference_path: str,
    current_data: pd.DataFrame,
    report_dir: str = "reports",
    reference_nrows: int = 10_000,
) -> dict:
    """
    Convenience function: run both data-drift and data-quality reports.
    Returns combined summary dict.
    """
    monitor = DataDriftMonitor(
        reference_path=reference_path,
        report_dir=report_dir,
        reference_nrows=reference_nrows,
    )
    return monitor.run(current_data)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evidently drift detection")
    parser.add_argument("--reference", required=True, help="Path to reference CSV (training data)")
    parser.add_argument("--current", required=True, help="Path to current/production CSV")
    parser.add_argument("--report-dir", default="reports", help="Directory to save HTML reports")
    parser.add_argument("--reference-nrows", type=int, default=10_000,
                        help="Max rows to load from reference")
    parser.add_argument("--current-nrows", type=int, default=None,
                        help="Max rows to load from current")
    args = parser.parse_args()

    monitor = DataDriftMonitor(
        reference_path=args.reference,
        report_dir=args.report_dir,
        reference_nrows=args.reference_nrows,
    )
    current_df = _load_and_prepare(args.current, nrows=args.current_nrows)
    summary = monitor.run(current_df)

    print(json.dumps(summary, indent=2))

    # Non-zero exit if drift detected (useful in CI to trigger alerts)
    if summary.get("dataset_drift"):
        logger.warning("⚠️  Dataset drift detected — consider retraining")
        sys.exit(1)
