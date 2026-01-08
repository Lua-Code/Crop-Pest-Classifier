#=======================================
# IMPORTS
#=======================================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import models

from src.data.data import DataConfig, getDatasetsAndFolds


#=============================
# CONFIG
#=============================
@dataclass(frozen=True)
class EmbeddingConfig:
    backbone: str = "resnet18" 
    batchSize: int = 32
    numWorkers: int = 2
    outputDir: Path = Path("runs")
    device: str = "auto" 
    
#=============================
# BACKBONE FEATURE EXTRACTOR
#=============================

#So i don't blow up your laptop lol
def getDevice(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def buildFeatureExtractor(backbone: str, device: torch.device) -> Tuple[nn.Module, int]:
    backbone = backbone.lower().strip()

    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        featureExtractor = nn.Sequential(*list(m.children())[:-1]) 
        embeddingDim = 512

    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        featureExtractor = nn.Sequential(
            m.features,
            m.avgpool,        
        )
        embeddingDim = 1280

    else:
        raise ValueError(f"Unsupported backbone '{backbone}'. Use 'resnet18' or 'efficientnet_b0'.")

    featureExtractor.to(device)
    featureExtractor.eval()
    for p in featureExtractor.parameters():
        p.requires_grad = False

    return featureExtractor, embeddingDim

#=============================
# EMBEDDING EXTRACTION
#=============================

@torch.no_grad()
def extractEmbeddings( dataset, featureExtractor: nn.Module, embeddingDim: int, device: torch.device, batchSize: int = 32, numWorkers: int = 2,) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=(device.type == "cuda"),
    )

    n = len(dataset)
    X = np.zeros((n, embeddingDim), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)

    offset = 0
    for images, labels in tqdm(loader, desc="Extracting embeddings", unit="batch"):
        images = images.to(device, non_blocking=True)

        feats = featureExtractor(images)

        feats = feats.view(feats.size(0), -1)

        b = feats.size(0)
        X[offset : offset + b] = feats.cpu().numpy().astype(np.float32)
        y[offset : offset + b] = labels.numpy().astype(np.int64)
        offset += b

    return X, y


def saveSplitEmbeddings(outputDir: Path, splitName: str, X: np.ndarray, y: np.ndarray) -> None:
    outputDir.mkdir(parents=True, exist_ok=True)
    np.save(outputDir / f"X_{splitName}.npy", X)
    np.save(outputDir / f"y_{splitName}.npy", y)
  
  
#=============================
# MAIN
#=============================   
       
def main():
    dataCfg = DataConfig(
        trainDir=Path("data/train"),
        testDir=Path("data/test"),
        trainCsv=Path("data/train.csv"),
        testCsv=Path("data/test.csv"),
        imageSize=224,
        seed=42,
        kFolds=5,
        shuffleFolds=True,
    )

    embCfg = EmbeddingConfig(
        backbone="resnet18",   
        batchSize=32,
        numWorkers=2,
        outputDir=Path("runs"),
        device="auto",
    )

    device = getDevice(embCfg.device)
    print(f"Using device: {device}")

    trainDs, testDs, folds, classToIdx = getDatasetsAndFolds(dataCfg)

    featureExtractor, embeddingDim = buildFeatureExtractor(embCfg.backbone, device)
    print(f"Backbone: {embCfg.backbone} | embeddingDim: {embeddingDim}")

    XTrain, yTrain = extractEmbeddings(
        dataset=trainDs,
        featureExtractor=featureExtractor,
        embeddingDim=embeddingDim,
        device=device,
        batchSize=embCfg.batchSize,
        numWorkers=embCfg.numWorkers,
    )
    saveSplitEmbeddings(embCfg.outputDir, "train", XTrain, yTrain)
    print("Saved:", embCfg.outputDir / "X_train.npy", "and", embCfg.outputDir / "y_train.npy")

    XTest, yTest = extractEmbeddings(
        dataset=testDs,
        featureExtractor=featureExtractor,
        embeddingDim=embeddingDim,
        device=device,
        batchSize=embCfg.batchSize,
        numWorkers=embCfg.numWorkers,
    )
    saveSplitEmbeddings(embCfg.outputDir, "test", XTest, yTest)
    print("Saved:", embCfg.outputDir / "X_test.npy", "and", embCfg.outputDir / "y_test.npy")

    print("XTrain shape:", XTrain.shape, "yTrain shape:", yTrain.shape)
    print("XTest shape:", XTest.shape, "yTest shape:", yTest.shape)


if __name__ == "__main__":
    main()