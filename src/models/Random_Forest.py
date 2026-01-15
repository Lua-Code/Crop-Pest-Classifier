from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from src.utils.metrics import (
    computeConfusionMatrix,
    computeMetrics,
    ensureDir,
    saveConfusionMatrixPlot,
    saveJson,
    summarizeCvResults,
)


# =============================
# Config
# =============================
@dataclass(frozen=True)
class RfConfig:
    xTrainPath: Path = Path("runs/X_train.npy")
    yTrainPath: Path = Path("runs/y_train.npy")
    xTestPath: Path = Path("runs/X_test.npy")
    yTestPath: Path = Path("runs/y_test.npy")

    outputDir: Path = Path("runs")
    seed: int = 42
    kFolds: int = 5

    
    nEstimators: int = 500
    maxDepth: int | None = None          
    minSamplesSplit: int = 2
    minSamplesLeaf: int = 1
    maxFeatures: str = "sqrt"           
    classWeight: str | None = "balanced" 
    nJobs: int = -1


# =============================
# Model builder
# =============================
def buildModel(cfg: RfConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.nEstimators,
        max_depth=cfg.maxDepth,
        min_samples_split=cfg.minSamplesSplit,
        min_samples_leaf=cfg.minSamplesLeaf,
        max_features=cfg.maxFeatures,
        class_weight=cfg.classWeight,
        random_state=cfg.seed,
        n_jobs=cfg.nJobs,
    )


# ==============================
# CV + Train + Test
# ==============================
def runCrossValidation(X: np.ndarray, y: np.ndarray, cfg: RfConfig) -> Dict:
    skf = StratifiedKFold(n_splits=cfg.kFolds, shuffle=True, random_state=cfg.seed)

    foldAccs: List[float] = []
    foldMacroF1s: List[float] = []
    foldWeightedF1s: List[float] = []

    splits = list(skf.split(X, y))
    for fold, (trIdx, vaIdx) in enumerate(tqdm(splits, desc="RF Cross-Validation", total=cfg.kFolds), start=1):
        model = buildModel(cfg)
        model.fit(X[trIdx], y[trIdx])
        pred = model.predict(X[vaIdx])

        m = computeMetrics(y[vaIdx], pred)
        foldAccs.append(m["accuracy"])
        foldMacroF1s.append(m["macroF1"])
        foldWeightedF1s.append(m["weightedF1"])

        tqdm.write(
            f"Fold {fold}/{cfg.kFolds} | acc={m['accuracy']:.4f} | macroF1={m['macroF1']:.4f} | weightedF1={m['weightedF1']:.4f}"
        )

    return {
        "kFolds": cfg.kFolds,
        "accuracy": summarizeCvResults(foldAccs),
        "macroF1": summarizeCvResults(foldMacroF1s),
        "weightedF1": summarizeCvResults(foldWeightedF1s),
    }


def main():
    cfg = RfConfig()
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

    # 2) Train final model on full training set
    print("\n=== Training final Random Forest on full train set ===")
    model = buildModel(cfg)
    model.fit(Xtrain, ytrain)

    # 3) Evaluate on test once
    print("\n=== Final evaluation on provided test set ===")
    testPred = model.predict(Xtest)
    testMetrics = computeMetrics(ytest, testPred)
    cm = computeConfusionMatrix(ytest, testPred)

    print(f"Test Accuracy : {testMetrics['accuracy']:.4f}")
    print(f"Test Macro F1 : {testMetrics['macroF1']:.4f}")
    print(f"Test Weighted : {testMetrics['weightedF1']:.4f}")

    # 4) Save artifacts
    modelPath = cfg.outputDir / "rf_model.joblib"
    metricsPath = cfg.outputDir / "rf_metrics.json"
    cmPath = cfg.outputDir / "rf_confusion_matrix.png"

    joblib.dump(model, modelPath)

    metricsOut = {
        "model": "RandomForestClassifier (embeddings)",
        "hyperparams": {
            "nEstimators": cfg.nEstimators,
            "maxDepth": cfg.maxDepth,
            "minSamplesSplit": cfg.minSamplesSplit,
            "minSamplesLeaf": cfg.minSamplesLeaf,
            "maxFeatures": cfg.maxFeatures,
            "classWeight": cfg.classWeight,
        },
        "cv": cv,
        "test": {
            "accuracy": testMetrics["accuracy"],
            "macroF1": testMetrics["macroF1"],
            "weightedF1": testMetrics["weightedF1"],
            "classificationReport": testMetrics["classificationReport"],
        },
    }

    saveJson(metricsOut, metricsPath)
    saveConfusionMatrixPlot(cm=cm, outPath=cmPath, title="Random Forest Confusion Matrix")

    print("\nSaved:")
    print(" ", modelPath)
    print(" ", metricsPath)
    print(" ", cmPath)


if __name__ == "__main__":
    main()
