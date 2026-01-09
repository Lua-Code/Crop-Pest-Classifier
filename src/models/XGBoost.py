#=============================
# XGBoost NDE (Fast + High Accuracy)
# GPU + CuPy + Early Stopping (XGBoost 3.1.2 sklearn wrapper compatible)
#=============================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import cupy as cp

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold

from src.utils.metrics import (
    computeConfusionMatrix,
    computeMetrics,
    saveConfusionMatrixPlot,
    saveJson,
    ensureDir,
)


#=============================
# CONFIG
#=============================
@dataclass(frozen=True)
class XgbNdeConfig:
    xTrainPath: Path = Path("runs/X_train.npy")
    yTrainPath: Path = Path("runs/y_train.npy")
    xTestPath: Path = Path("runs/X_test.npy")
    yTestPath: Path = Path("runs/y_test.npy")

    outputDir: Path = Path("runs")
    seed: int = 42

    # GPU
    gpuId: int = 0

    # Early stopping (only used in stage 1)
    nEstimatorsCap: int = 3000
    earlyStoppingRounds: int = 50
    valFolds: int = 10  # 1/10 validation split

    # Strong fast defaults for embeddings
    maxDepth: int = 5
    learningRate: float = 0.05
    subsample: float = 0.9
    colsampleBytree: float = 0.9
    minChildWeight: int = 1
    regLambda: float = 1.0
    regAlpha: float = 0.0
    gamma: float = 0.0

    # Plot settings
    cmNormalize: str | None = None


#=============================
# HELPERS
#=============================
def to_gpu(X: np.ndarray) -> "cp.ndarray":
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return cp.asarray(X)


def gpu_params(cfg: XgbNdeConfig) -> Dict[str, Any]:
    return {"device": f"cuda:{cfg.gpuId}", "tree_method": "hist"}


def build_model(
    cfg: XgbNdeConfig,
    numClasses: int,
    n_estimators: int,
    early_stopping_rounds: Optional[int] = None,
) -> XGBClassifier:
    """
    For XGBoost 3.1.2:
    - early_stopping_rounds must be passed to the constructor (not fit, not callbacks)
    - if early_stopping_rounds is set, fit MUST have eval_set
    """
    kwargs = {}
    if early_stopping_rounds is not None:
        kwargs["early_stopping_rounds"] = int(early_stopping_rounds)

    return XGBClassifier(
        objective="multi:softprob",
        num_class=numClasses,

        n_estimators=int(n_estimators),
        learning_rate=cfg.learningRate,
        max_depth=cfg.maxDepth,

        subsample=cfg.subsample,
        colsample_bytree=cfg.colsampleBytree,

        min_child_weight=cfg.minChildWeight,
        reg_lambda=cfg.regLambda,
        reg_alpha=cfg.regAlpha,
        gamma=cfg.gamma,

        eval_metric="mlogloss",
        random_state=cfg.seed,
        n_jobs=-1,

        **kwargs,
        **gpu_params(cfg),
    )


#=============================
# MAIN
#=============================
def main():
    cfg = XgbNdeConfig()
    ensureDir(cfg.outputDir)

    print("XGBoost version:", xgb.__version__)
    print("GPU:", f"cuda:{cfg.gpuId}")

    Xtrain = np.load(cfg.xTrainPath)
    ytrain = np.load(cfg.yTrainPath)
    Xtest = np.load(cfg.xTestPath)
    ytest = np.load(cfg.yTestPath)

    print("Loaded:")
    print("  Xtrain:", Xtrain.shape, Xtrain.dtype)
    print("  ytrain:", ytrain.shape, ytrain.dtype)
    print("  Xtest :", Xtest.shape, Xtest.dtype)
    print("  ytest :", ytest.shape, ytest.dtype)

    numClasses = int(np.unique(ytrain).size)

    # 1) Stratified validation split (fast)
    skf = StratifiedKFold(n_splits=cfg.valFolds, shuffle=True, random_state=cfg.seed)
    trIdx, vaIdx = next(skf.split(Xtrain, ytrain))

    X_tr = to_gpu(Xtrain[trIdx])
    y_tr = ytrain[trIdx]
    X_va = to_gpu(Xtrain[vaIdx])
    y_va = ytrain[vaIdx]

    # 2) Stage 1: Train with early stopping to find best trees
    print("\n=== Training (GPU + CuPy + EarlyStopping) ===")
    model_es = build_model(
        cfg,
        numClasses=numClasses,
        n_estimators=cfg.nEstimatorsCap,
        early_stopping_rounds=cfg.earlyStoppingRounds,
    )
    model_es.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    best_iter = getattr(model_es, "best_iteration", None)
    best_trees = (int(best_iter) + 1) if best_iter is not None else cfg.nEstimatorsCap
    print("Best iteration (trees):", best_trees)

    # 3) Stage 2: Refit on full train WITHOUT early stopping
    print("\n=== Refit on full train with best trees ===")
    final_model = build_model(
        cfg,
        numClasses=numClasses,
        n_estimators=best_trees,
        early_stopping_rounds=None,   # ✅ disable early stopping here
    )

    Xtrain_gpu = to_gpu(Xtrain)
    final_model.fit(Xtrain_gpu, ytrain, verbose=False)

    # 4) Evaluate on test
    print("\n=== Final evaluation on provided test set ===")
    Xtest_gpu = to_gpu(Xtest)
    testPred = final_model.predict(Xtest_gpu)

    testMetrics = computeMetrics(ytest, testPred)
    print(f"Test Accuracy : {testMetrics['accuracy']:.4f}")
    print(f"Test Macro F1 : {testMetrics['macroF1']:.4f}")
    print(f"Test Weighted : {testMetrics['weightedF1']:.4f}")

    cm = computeConfusionMatrix(ytest, testPred)

    # 5) Save artifacts
    modelPath = cfg.outputDir / "xgb_nde_model.joblib"
    metricsPath = cfg.outputDir / "xgb_nde_metrics.json"
    cmPath = cfg.outputDir / "xgb_nde_confusion_matrix.png"

    joblib.dump(final_model, modelPath)

    metricsOut = {
        "model": "XGBoost NDE (fast defaults + early stopping, GPU + CuPy)",
        "xgboostVersion": xgb.__version__,
        "gpu": f"cuda:{cfg.gpuId}",
        "bestTrees": int(best_trees),
        "config": {
            "maxDepth": cfg.maxDepth,
            "learningRate": cfg.learningRate,
            "subsample": cfg.subsample,
            "colsampleBytree": cfg.colsampleBytree,
            "minChildWeight": cfg.minChildWeight,
            "regLambda": cfg.regLambda,
            "regAlpha": cfg.regAlpha,
            "gamma": cfg.gamma,
            "earlyStoppingRounds": cfg.earlyStoppingRounds,
            "valFolds": cfg.valFolds,
            "nEstimatorsCap": cfg.nEstimatorsCap,
        },
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
        title="XGBoost NDE Confusion Matrix (GPU + CuPy)",
        normalize=cfg.cmNormalize,
    )

    print("\nSaved:")
    print(" ", modelPath)
    print(" ", metricsPath)
    print(" ", cmPath)


if __name__ == "__main__":
    main()
