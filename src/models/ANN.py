from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.metrics import (
    computeConfusionMatrix,
    computeMetrics,
    ensureDir,
    saveConfusionMatrixPlot,
    saveJson,
    summarizeCvResults,
)


@dataclass(frozen=True)
class AnnConfig:
    xTrainPath: Path = Path("runs/X_train.npy")
    yTrainPath: Path = Path("runs/y_train.npy")
    xTestPath: Path = Path("runs/X_test.npy")
    yTestPath: Path = Path("runs/y_test.npy")

    outputDir: Path = Path("runs")
    seed: int = 42
    kFolds: int = 5

    hiddenLayerSizes: tuple[int, int] = (256, 128)
    alpha: float = 1e-4                 
    learningRateInit: float = 1e-3
    maxIter: int = 200                  
    earlyStopping: bool = True
    validationFraction: float = 0.1
    nIterNoChange: int = 10


def buildModel(cfg: AnnConfig) -> Pipeline:
    mlp = MLPClassifier(
        hidden_layer_sizes=cfg.hiddenLayerSizes,
        activation="relu",
        solver="adam",
        alpha=cfg.alpha,
        learning_rate_init=cfg.learningRateInit,
        max_iter=cfg.maxIter,
        early_stopping=cfg.earlyStopping,
        validation_fraction=cfg.validationFraction,
        n_iter_no_change=cfg.nIterNoChange,
        random_state=cfg.seed,
    )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", mlp),
        ]
    )


def runCrossValidation(X: np.ndarray, y: np.ndarray, cfg: AnnConfig) -> Dict:
    skf = StratifiedKFold(n_splits=cfg.kFolds, shuffle=True, random_state=cfg.seed)

    accs: List[float] = []
    macroF1s: List[float] = []
    weightedF1s: List[float] = []

    splits = list(skf.split(X, y))
    for fold, (trIdx, vaIdx) in enumerate(tqdm(splits, desc="ANN (sklearn) CV", total=cfg.kFolds), start=1):
        model = buildModel(cfg)
        model.fit(X[trIdx], y[trIdx])
        pred = model.predict(X[vaIdx])

        m = computeMetrics(y[vaIdx], pred)
        accs.append(m["accuracy"])
        macroF1s.append(m["macroF1"])
        weightedF1s.append(m["weightedF1"])

        tqdm.write(
            f"Fold {fold}/{cfg.kFolds} | acc={m['accuracy']:.4f} | macroF1={m['macroF1']:.4f} | weightedF1={m['weightedF1']:.4f}"
        )

    return {
        "kFolds": cfg.kFolds,
        "accuracy": summarizeCvResults(accs),
        "macroF1": summarizeCvResults(macroF1s),
        "weightedF1": summarizeCvResults(weightedF1s),
    }


def main():
    cfg = AnnConfig()
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

    print("\n=== Stratified K-Fold CV (train only) ===")
    cv = runCrossValidation(Xtrain, ytrain, cfg)
    print(
        "\nCV Summary:\n"
        f"  Accuracy : {cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.4f}\n"
        f"  Macro F1 : {cv['macroF1']['mean']:.4f} ± {cv['macroF1']['std']:.4f}\n"
        f"  Weighted : {cv['weightedF1']['mean']:.4f} ± {cv['weightedF1']['std']:.4f}"
    )

    print("\n=== Training final ANN on full train set ===")
    model = buildModel(cfg)
    model.fit(Xtrain, ytrain)

    print("\n=== Final evaluation on provided test set ===")
    testPred = model.predict(Xtest)
    testMetrics = computeMetrics(ytest, testPred)
    cm = computeConfusionMatrix(ytest, testPred)

    print(f"Test Accuracy : {testMetrics['accuracy']:.4f}")
    print(f"Test Macro F1 : {testMetrics['macroF1']:.4f}")
    print(f"Test Weighted : {testMetrics['weightedF1']:.4f}")

    modelPath = cfg.outputDir / "ann_model.joblib"
    metricsPath = cfg.outputDir / "ann_metrics.json"
    cmPath = cfg.outputDir / "ann_confusion_matrix.png"

    joblib.dump(model, modelPath)

    metricsOut = {
        "model": "StandardScaler + sklearn.MLPClassifier",
        "hyperparams": {
            "hiddenLayerSizes": cfg.hiddenLayerSizes,
            "alpha": cfg.alpha,
            "learningRateInit": cfg.learningRateInit,
            "maxIter": cfg.maxIter,
            "earlyStopping": cfg.earlyStopping,
            "validationFraction": cfg.validationFraction,
            "nIterNoChange": cfg.nIterNoChange,
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
    saveConfusionMatrixPlot(cm=cm, outPath=cmPath, title="ANN Confusion Matrix")

    print("\nSaved:")
    print(" ", modelPath)
    print(" ", metricsPath)
    print(" ", cmPath)


if __name__ == "__main__":
    main()
