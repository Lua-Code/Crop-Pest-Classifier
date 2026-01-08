from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)



def ensureDir(dirPath: Path) -> None:
    dirPath.mkdir(parents=True, exist_ok=True)


def computeMetrics(
    yTrue: Sequence[int],
    yPred: Sequence[int],
    labels: Optional[Sequence[int]] = None,
    classNames: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    
    acc = float(accuracy_score(yTrue, yPred))
    macroF1 = float(f1_score(yTrue, yPred, average="macro"))
    weightedF1 = float(f1_score(yTrue, yPred, average="weighted"))

    report = classification_report(
        yTrue,
        yPred,
        labels=labels,
        target_names=classNames,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macroF1": macroF1,
        "weightedF1": weightedF1,
        "classificationReport": report,
    }


def computeConfusionMatrix(
    yTrue: Sequence[int],
    yPred: Sequence[int],
    labels: Optional[Sequence[int]] = None,
) -> np.ndarray:
    cm = confusion_matrix(yTrue, yPred, labels=labels)
    return cm


def saveConfusionMatrixPlot(
    cm: np.ndarray,
    outPath: Path,
    classNames: Optional[Sequence[str]] = None,
    title: str = "Confusion Matrix",
    dpi: int = 200,
    normalize: Optional[str] = None,  # None, "true", "pred", "all"
) -> None:
    ensureDir(outPath.parent)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classNames)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(
        ax=ax,
        xticks_rotation=45,
        values_format=".2f" if normalize else "d",
        colorbar=False,
    )

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outPath, dpi=dpi)
    plt.close(fig)

#useless for now tbh but ill leave them just incase
def saveJson(data: Dict[str, Any], outPath: Path) -> None:
    ensureDir(outPath.parent)
    outPath.write_text(json.dumps(data, indent=2))


def summarizeCvResults(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": mean, "std": std}
