#=======================================
# IMPORTS
#=======================================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold

#lets configure our data settings >:3
@dataclass(frozen=True)
class DataConfig:
    trainDir: Path
    testDir: Path
    trainCsv: Path
    testCsv: Path
    imageSize: int = 224
    seed: int = 42
    kFolds: int = 5
    shuffleFolds: bool = True
    

#=======================================
# HELPERS
#=======================================

# Find column in dataframe matching any of the candidate names (case insensitive)
def findColumn(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    lowerToReal = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lowerToReal:
            return lowerToReal[key]
    raise ValueError(
        f"Could not find any of {list(candidates)} in CSV columns: {list(df.columns)}"
    )
    
# Since this a kaggle dataset we need to normalize the paths    
def normalizePath(pathStr: str) -> str:
    p = pathStr.replace("\\", "/").strip()

    marker = "agricultural-pests-image-dataset/"
    if marker in p:
        p = p.split(marker, 1)[1]

    return p.lstrip("/")

# Load metadata from CSV and return a DataFrame with 'filename' and 'label' columns
def loadMetadata(csvPath: Path) -> pd.DataFrame:
    df = pd.read_csv(csvPath)

    filenameCol = findColumn(df, ["filename", "file", "image", "img", "path"])
    labelCol = findColumn(df, ["label", "class", "category", "target"])

    out = df[[filenameCol, labelCol]].copy()
    out.columns = ["filename", "label"]

    out["filename"] = out["filename"].astype(str).apply(normalizePath)
    out["label"] = out["label"].astype(str).str.strip()

    return out

# Manual Label Encoding, returns class name to index mapping
def buildClassMapping(trainDf: pd.DataFrame) -> Dict[str, int]:
    classes = sorted(trainDf["label"].unique().tolist())
    return {clsName: idx for idx, clsName in enumerate(classes)}


def resolveImagePath(baseDir: Path, filename: str, label: Optional[str] = None) -> Path:
    p1 = baseDir / filename
    if p1.exists():
        return p1

    if label is not None:
        p2 = baseDir / label / filename
        if p2.exists():
            return p2

    matches = list(baseDir.rglob(Path(filename).name))
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not resolve image path for filename='{filename}' (label='{label}') "
        f"under baseDir='{baseDir}'. Tried '{p1}' and '{baseDir/str(label)/filename}'."
    )
    
# ==============================
# Dataset (returns (imageTensor, labelInt))
#================================

#We will be using a pytorch dataset to load our images and labels, keep that in mind please ya meligy w hania :3
class PestImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        imagesDir: Path,
        classToIdx: Dict[str, int],
        transform=None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.imagesDir = imagesDir
        self.classToIdx = classToIdx
        self.transform = transform

        self.y: List[int] = [self.classToIdx[lbl] for lbl in self.df["label"].tolist()]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        filename = row["filename"]
        labelStr = row["label"]
        labelInt = self.classToIdx[labelStr]

        imgPath = resolveImagePath(self.imagesDir, filename, labelStr)
        img = Image.open(imgPath).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, labelInt
    
#=======================================
# PREPROCESSING
#=======================================

def buildTransforms(imageSize: int = 224):
    imagenetMean = (0.485, 0.456, 0.406)
    imagenetStd = (0.229, 0.224, 0.225)

    trainTransforms = transforms.Compose(
        [
            transforms.Resize((imageSize, imageSize)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(imagenetMean, imagenetStd),
        ]
    )

    evalTransforms = transforms.Compose(
        [
            transforms.Resize((imageSize, imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(imagenetMean, imagenetStd),
        ]
    )

    return trainTransforms, evalTransforms

#=======================================
# K-FOLD SPLITTING(TRAIN)
#=======================================

def makeTrainFolds( yTrain: Sequence[int], kFolds: int = 5, seed: int = 42, shuffle: bool = True,) -> List[Tuple[List[int], List[int]]]:
    skf = StratifiedKFold(n_splits=kFolds, shuffle=shuffle, random_state=seed)
    folds: List[Tuple[List[int], List[int]]] = []
    yTrain = list(yTrain)

    for trIdx, vaIdx in skf.split(X=[0] * len(yTrain), y=yTrain):
        folds.append((trIdx.tolist(), vaIdx.tolist()))

    return folds

#=======================================
# WRAPPER
#=======================================

def getDatasetsAndFolds(cfg: DataConfig):
    trainDf = loadMetadata(cfg.trainCsv)
    testDf = loadMetadata(cfg.testCsv)

    classToIdx = buildClassMapping(trainDf)

    unknown = sorted(set(testDf["label"].unique()) - set(classToIdx.keys()))
    if unknown:
        raise ValueError(
            f"Labels in test.csv not present in train.csv: {unknown}. "
            f"Train classes: {sorted(classToIdx.keys())}"
        )

    trainTransforms, evalTransforms = buildTransforms(cfg.imageSize)

    trainDs = PestImageDataset(trainDf, cfg.trainDir, classToIdx, transform=trainTransforms)
    testDs = PestImageDataset(testDf, cfg.testDir, classToIdx, transform=evalTransforms)

    folds = makeTrainFolds(
        yTrain=trainDs.y,
        kFolds=cfg.kFolds,
        seed=cfg.seed,
        shuffle=cfg.shuffleFolds,
    )

    return trainDs, testDs, folds, classToIdx



if __name__ == "__main__":
    cfg = DataConfig(
        trainDir=Path("data/train"),
        testDir=Path("data/test"),
        trainCsv=Path("data/train.csv"),
        testCsv=Path("data/test.csv"),
        imageSize=224,
        seed=42,
        kFolds=5,
        shuffleFolds=True,
    )

    trainDs, testDs, folds, classToIdx = getDatasetsAndFolds(cfg)

    print("Train samples:", len(trainDs))
    print("Test samples:", len(testDs))
    print("Num classes:", len(classToIdx))

    x, y = trainDs[0]
    print("Sample tensor shape:", tuple(x.shape), "label int:", y)

    print("Fold 1 sizes:", len(folds[0][0]), "(train)", len(folds[0][1]), "(val)")