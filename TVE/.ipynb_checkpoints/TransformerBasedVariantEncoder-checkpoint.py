import os
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import warnings
import sys
import time

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)




def setSeed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def createCausalMask(seqLen: int) -> torch.Tensor:
    """Create a causal mask for transformer."""
    return torch.triu(torch.ones(seqLen, seqLen, dtype=torch.bool), diagonal=1)


def getActivationFunc(name: str):
    """Return the corresponding activation function."""
    if name == "relu":
        return F.relu
    elif name == "gelu":
        return F.gelu
    elif name == "glu":
        return F.glu
    else:
        raise ValueError(f"Unsupported activation: {name}")


class PositionalEncoding(nn.Module):
    """Injects positional information into the input tensor."""

    def __init__(self, dimModel, dropout=0.1, maxLen=12279):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        posEncoding = torch.zeros(maxLen, dimModel)
        position = torch.arange(0, maxLen).unsqueeze(1).float()
        divTerm = torch.exp(torch.arange(0, dimModel, 2).float() * (-math.log(10000.0) / dimModel))

        posEncoding[:, 0::2] = torch.sin(position * divTerm)
        if dimModel % 2 == 0:
            posEncoding[:, 1::2] = torch.cos(position * divTerm)
        else:
            posEncoding[:, 1::2] = torch.cos(position * divTerm)[:, :-1]

        posEncoding = posEncoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('posEncoding', posEncoding)

    def forward(self, x):
        x = x + self.posEncoding[:x.size(0)].squeeze(1)
        return self.dropout(x)


class TransformerVariantEncoder(nn.Module):
    """Custom Transformer-based encoder for variant representations."""

    def __init__(self, numLayers=1, dimModel=15, numHeads=3, dimFeedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()
        self.positionalEncoding = PositionalEncoding(dimModel, dropout)
        encoderLayer = nn.TransformerEncoderLayer(dimModel, numHeads, dimFeedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(encoderLayer, numLayers)

    def forward(self, src, srcMask=None):
        src = self.positionalEncoding(src)
        return self.encoder(src, srcMask)


class TransformerBasedVariantEncoderRunner:
    """
    Wrapper to run the TransformerVariantEncoder on a CSV input and save outputs.
    """

    def __init__(self, dataPath: str, modelDir: str, outputDir: str, dimModel: int, numHeads: int = 5):
        self.dataPath = dataPath
        self.modelDir = modelDir
        self.outputDir = outputDir
        self.dimModel = dimModel
        self.numHeads = numHeads

        os.makedirs(modelDir, exist_ok=True)
        os.makedirs(outputDir, exist_ok=True)

        df = pd.read_csv(self.dataPath, dtype=str)
        featCols = df.columns[0:df.columns.get_loc('FA15') + 1]
        self.featureData = df[featCols].astype(float).values.astype(np.float32)
        self.metaData = df

    def run(self, maxLayers: int = 20):
        print(f"Starting the encoding process with {maxLayers} layers...")
        print("=" * 100)  # Divider for better visibility

        red_square = '\033[31m■\033[0m'  # ANSI escape code for red color
        empty_square = '□'  # Empty square to represent uncompleted progress

        for layer in range(1, maxLayers + 1):
            # Manually update the progress bar
            progress = (layer / maxLayers) * 100
            bar_length = 40  # Length of the progress bar
            block = int(round(bar_length * progress / 100))
            progress_bar = f"Processing Layers: [{'■' * block}{'□' * (bar_length - block)}] {progress:.2f}%"

            # Print the progress bar
            sys.stdout.write(f"\r{progress_bar}")
            sys.stdout.flush()

            # Print the other info without interfering with the progress bar
            print(
                f"\n----------------------------------------------------------------------------------------------------")
            print(f"Processing layer {layer}/{maxLayers}...")

            model = TransformerVariantEncoder(
                numLayers=layer,
                dimModel=self.dimModel,
                numHeads=self.numHeads
            )
            modelPath = os.path.join(self.modelDir, f'encoderLayer{layer}.pth')
            torch.save(model.state_dict(), modelPath)
            print(f"[1/4] Model for layer {layer} saved at: {modelPath}")

            # Prepare input tensor and mask
            inputTensor = torch.tensor(self.featureData)
            mask = createCausalMask(inputTensor.size(0))

            print(f"[2/4] Encoding the input tensor using the transformer model...")
            encoded = model(inputTensor, srcMask=mask).detach().numpy()

            print(f"[3/4] Processing the encoded output...")
            encodedFrame = pd.DataFrame(encoded)
            metaCols = ['e_total', 'e_ionic', 'e_electronic', 'formula_pretty', 'Material ID']
            outputFrame = pd.concat([encodedFrame, self.metaData[metaCols]], axis=1)

            outputPath = os.path.join(self.outputDir, f'encodedOutputLayer{layer}.csv')
            outputFrame.to_csv(outputPath, index=False)

            print(f"[4/4] Layer {layer} encoding completed. Output saved to: {outputPath}")
            print("\n")

            # Clear memory for the next iteration
            del model, inputTensor, encoded

        print("=" * 100)
        print(f"All encoding processes completed. All outputs have been saved to '{self.outputDir}'.")
        print(f"Total of {maxLayers} layers processed.")
