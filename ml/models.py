"""CNN архитектуралары: өзіндік PlantCNN және MobileNetV2 (transfer learning)."""
import torch
import torch.nn as nn
from torchvision import models


def conv_block(c_in, c_out):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


WIDTHS = (16, 32, 64, 128)  # PlantCNN блоктарындағы арналар саны
HIDDEN = 128                 # жасырын толық байланысқан қабат өлшемі


class PlantCNN(nn.Module):
    """Нөлден оқытылатын 8 конволюциялық қабатты CNN.
    Кіріс: 3x128x128 -> 4 блок (16-32-64-128) -> GAP -> Dropout -> FC -> FC."""

    def __init__(self, num_classes: int):
        super().__init__()
        chans = (3,) + WIDTHS
        self.features = nn.Sequential(*[conv_block(chans[i], chans[i + 1]) for i in range(len(WIDTHS))])
        # 128 -> 64 -> 32 -> 16 -> 8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(WIDTHS[-1], HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class BaselineCNN(nn.Module):
    """Three-block coursework baseline; optional BN/Dropout experiment.

    This architecture is separate from the existing eight-convolution PlantCNN
    checkpoint. Adding it never reinterprets or replaces those saved weights.
    """
    def __init__(self, num_classes: int, regularized: bool = False):
        super().__init__()
        blocks = []
        channels = (3, 32, 64, 128)
        for i in range(3):
            layers = [nn.Conv2d(channels[i], channels[i + 1], 3, padding=1)]
            if regularized:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=False))
            if i < 2:
                layers.append(nn.MaxPool2d(2))
            blocks.append(nn.Sequential(*layers))
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(.3) if regularized else nn.Identity(), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def mobilenet_v2(num_classes: int, pretrained: bool = True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.mobilenet_v2(weights=weights)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, num_classes))
    return m


MODEL_SPECS = {
    "plantcnn": {"title": "PlantCNN (өзіндік CNN)", "img_size": 128},
    "mobilenet": {"title": "MobileNetV2 (transfer learning)", "img_size": 192},
    "baseline": {"title": "Baseline CNN (3 блок)", "img_size": 128},
    "regularized": {"title": "Regularized CNN (BN + Dropout)", "img_size": 128},
}


def build_model(name: str, num_classes: int, pretrained: bool = False):
    if name in {"baseline", "regularized"}:
        return BaselineCNN(num_classes, regularized=name == "regularized")
    if name == "plantcnn":
        return PlantCNN(num_classes)
    if name == "mobilenet":
        return mobilenet_v2(num_classes, pretrained)
    raise ValueError(name)


def gradcam_target_layer(model, name: str):
    if name == "plantcnn":
        return model.features[-1][5]  # соңғы ReLU
    return model.features[-1]          # MobileNetV2 соңғы Conv-BN-ReLU6


def count_params(model):
    return sum(p.numel() for p in model.parameters())


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
