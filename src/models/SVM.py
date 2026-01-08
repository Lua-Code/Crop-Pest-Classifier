#=============================
# SVM Model
#=============================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from tqdm import tqdm

from src.utils.metrics import (
    computeConfusionMatrix,
    computeMetrics,
    saveConfusionMatrixPlot,
    saveJson,
    summarizeCvResults,
    ensureDir,
)


#=============================
# CONFIG
#=============================
@dataclass(frozen=True)
class SvmConfig:
    xTrainPath: Path = Path("runs/X_train.npy")
    yTrainPath: Path = Path("runs/y_train.npy")
    xTestPath: Path = Path("runs/X_test.npy")
    yTestPath: Path = Path("runs/y_test.npy")

    outputDir: Path = Path("runs")
    seed: int = 42
    kFolds: int = 5

    # Linear SVM hyperparams
    C: float = 1.0
    maxIter: int = 20000

    # Plot settings
    cmNormalize: str | None = None  # None, "true", "pred", "all"


#=============================
# MODEL BUILDING
#=============================

#our simple af pipeline
def buildModel(cfg: SvmConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(C=cfg.C, max_iter=cfg.maxIter, random_state=cfg.seed)),
        ]
    )


# =============================
# CROSS-VALIDATION
# =============================
def runCrossValidation(X: np.ndarray, y: np.ndarray, cfg: SvmConfig) -> Dict:
    skf = StratifiedKFold(n_splits=cfg.kFolds, shuffle=True, random_state=cfg.seed)

    foldAccs: List[float] = []
    foldMacroF1s: List[float] = []
    foldWeightedF1s: List[float] = []

    for fold, (trIdx, vaIdx) in enumerate(skf.split(X, y), start=1):
        Xtr, ytr = X[trIdx], y[trIdx]
        Xva, yva = X[vaIdx], y[vaIdx]

        model = buildModel(cfg)
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)

        m = computeMetrics(yva, pred)
        foldAccs.append(m["accuracy"])
        foldMacroF1s.append(m["macroF1"])
        foldWeightedF1s.append(m["weightedF1"])

        print(
            f"Fold {fold}/{cfg.kFolds} | "
            f"acc={m['accuracy']:.4f} | macroF1={m['macroF1']:.4f} | weightedF1={m['weightedF1']:.4f}"
        )

    cvResults = {
        "kFolds": cfg.kFolds,
        "accuracy": summarizeCvResults(foldAccs),
        "macroF1": summarizeCvResults(foldMacroF1s),
        "weightedF1": summarizeCvResults(foldWeightedF1s),
    }
    return cvResults


def main():
    cfg = SvmConfig()
    ensureDir(cfg.outputDir)

    Xtrain = np.load(cfg.xTrainPath)
    ytrain = np.load(cfg.yTrainPath)
    Xtest = np.load(cfg.xTestPath)
    ytest = np.load(cfg.yTestPath)

    print("Loaded:")
    print("  Xtrain:", Xtrain.shape, Xtrain.dtype)
    print("  ytrain:", ytrain.shape, ytrain.dtype)
    print("  Xtest :", Xtest.shape, Xtest.dtype)
    print("  ytest :", ytest.shape, ytest.dtype)

    # 1) Cross-validation on train only
    print("\n=== Stratified K-Fold CV (train only) ===")
    cv = runCrossValidation(Xtrain, ytrain, cfg)
    print(
        "\nCV Summary:\n"
        f"  Accuracy : {cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.4f}\n"
        f"  Macro F1 : {cv['macroF1']['mean']:.4f} ± {cv['macroF1']['std']:.4f}\n"
        f"  Weighted : {cv['weightedF1']['mean']:.4f} ± {cv['weightedF1']['std']:.4f}"
    )

    # 2) Train final model on full train set
    print("\n=== Training final SVM on full train set ===")
    model = buildModel(cfg)
    model.fit(Xtrain, ytrain)

    # 3) Evaluate on test once
    print("\n=== Final evaluation on provided test set ===")
    testPred = model.predict(Xtest)
    testMetrics = computeMetrics(ytest, testPred)
    print(f"Test Accuracy : {testMetrics['accuracy']:.4f}")
    print(f"Test Macro F1 : {testMetrics['macroF1']:.4f}")
    print(f"Test Weighted : {testMetrics['weightedF1']:.4f}")

    cm = computeConfusionMatrix(ytest, testPred)

    modelPath = cfg.outputDir / "svm_model.joblib"
    metricsPath = cfg.outputDir / "svm_metrics.json"
    cmPath = cfg.outputDir / "svm_confusion_matrix.png"

    joblib.dump(model, modelPath)

    metricsOut = {
        "model": "StandardScaler + LinearSVC",
        "hyperparams": {"C": cfg.C, "maxIter": cfg.maxIter},
        "cv": cv,
        "test": {
            "accuracy": testMetrics["accuracy"],
            "macroF1": testMetrics["macroF1"],
            "weightedF1": testMetrics["weightedF1"],
            "classificationReport": testMetrics["classificationReport"],
        },
    }

    saveJson(metricsOut, metricsPath)
    saveConfusionMatrixPlot(
        cm=cm,
        outPath=cmPath,
        title="SVM Confusion Matrix",
        normalize=cfg.cmNormalize,
    )

    print("\nSaved:")
    print(" ", modelPath)
    print(" ", metricsPath)
    print(" ", cmPath)


if __name__ == "__main__":
    main()
