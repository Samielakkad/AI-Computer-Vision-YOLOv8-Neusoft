import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    # kernel, padding, dilation
    # Calcule automatiquement le padding "same" pour que la feature map
    # garde la même résolution spatiale après convolution (stride=1).
    if d > 1:
        # Dilation dilate le kernel effectif : k_eff = d*(k-1)+1
        k = d * (k - 1) + 1 if isinstance(k, int) else [id * (x - 1) + 1 for x in k]
    if p is None:
        if isinstance(k, int):
            p = k // 2
        elif isinstance(k, tuple):
            p = [x // 2 for x in k]
        else:
            raise TypeError(f"Unsupported type for k: {type(k)}")
    return p


class SiLu(nn.Module):
    # Activation SiLU (Sigmoid Linear Unit) : x * sigmoid(x).
    # Préférée à ReLU dans YOLOv8 — lisse, non-monotone, légèrement
    # négative pour x < 0, ce qui préserve plus d'information de gradient.
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    # Brique de base de tout le réseau : Conv2d + BatchNorm2d + SiLU.
    # Le BN normalise les activations entre batches, ce qui stabilise
    # l'entraînement et permet des learning rates plus élevés.
    # forward_fuse() bypasse le BN après fusion à l'inférence.
    default_act = SiLu()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(
            c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True
        )
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # Après fuse() : le BN est absorbé dans les poids Conv,
        # on saute la couche BN pour gagner du temps à l'inférence.
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Bottleneck résiduel standard : deux Conv 3×3 avec expansion e=0.5
    # sur les channels internes, puis addition skip si c1==c2.
    # c1 = channels entrants, c2 = channels sortants.
    # Le shortcut permet au gradient de remonter directement — clé
    # pour entraîner des backbones profonds sans vanishing gradient.
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    # C2f = Cross Stage Partial with 2 convolutions + n Bottlenecks en séquence.
    # c1 = channels entrants, c2 = channels sortants.
    #
    # Idée centrale : cv1 projette vers 2*c channels, on splitte en deux moitiés.
    # Chaque Bottleneck traite la moitié droite et on empile (cat) tous les
    # résultats intermédiaires avant la projection finale cv2.
    # Ce "split-and-stack" crée des chemins de gradient multiples — équivalent
    # à un dense residual léger, sans le coût mémoire d'un DenseNet complet.
    def __init__(self, c1, c2, n=1, short_cut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, short_cut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        # cv1 projette puis split en deux tenseurs de c channels chacun
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # Chaque Bottleneck enrichit y[-1] ; on conserve TOUTES les sorties
        # intermédiaires — c'est le dense residual qui améliore le gradient flow.
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    # SPPF = Spatial Pyramid Pooling Fast : trois MaxPool 5×5 empilés en série.
    # Un MaxPool 5×5 appliqué 3 fois couvre des champs récepteurs effectifs
    # de 5, 9 et 13 pixels — même effet que trois poolings parallèles SPP,
    # mais deux fois plus rapide en pratique.
    # On concatène x, y1, y2, y3 pour avoir l'information multi-échelle
    # en un seul tenseur c_*4 channels, puis on reprojette vers c2.
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Backbone(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        # -----------------------
        # Image d'entrée : 3, 640, 640
        # -----------------------

        # stem : downsample x2 — 3, 640, 640 => 32, 640, 640 => base_channels, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)

        # dark2 : downsample x2 + C2f — base_channels, 320, 320 => base_channels*2, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # dark3 : downsample x2 + C2f — base_channels*2, 160, 160 => 256, 80, 80
        # stride 8 par rapport à l'image d'entrée — détecte les petits objets
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # dark4 : downsample x2 + C2f — 256, 80, 80 => 512, 40, 40
        # stride 16 — objets de taille moyenne
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
        )
        # dark5 : downsample x2 + C2f + SPPF — 512, 40, 40 => 1024*deep_mul, 20, 20
        # stride 32 — grands objets ; SPPF agrandit le champ récepteur ici
        # sans perdre encore de résolution spatiale
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), 2),
            SPPF(
                int(base_channels * 16 * deep_mul),
                int(base_channels * 16 * deep_mul),
                k=5,
            ),
        )

        if pretrained:
            url = {
                "n": "./pth/yolov8_n_backbone_weights.pth",
                "s": "./pth/yolov8_s_backbone_weights.pth",
                "m": "./pth/yolov8_m_backbone_weights.pth",
                "l": "./pth/yolov8_l_backbone_weights.pth",
                "x": "./pth/yolov8_x_backbone_weights.pth",
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", model_dir="./model_data"
            )
            self.load_state_dict(checkpoint, strict=False)
            print(f"Chargement des poids depuis {url.split('/')[-1]}")

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        # ---------------------------------
        # feat1 : sortie dark3 — 256, 80, 80 — feature map stride 8
        # utilisée pour détecter les petits objets dans le neck
        # ---------------------------------
        x = self.dark3(x)
        feat1 = x
        # ---------------------------------
        # feat2 : sortie dark4 — 512, 40, 40 — feature map stride 16
        # résolution intermédiaire, objets de taille moyenne
        # ---------------------------------
        x = self.dark4(x)
        feat2 = x

        # ---------------------------------
        # feat3 : sortie dark5 — 1024*deep_mul, 20, 20 — feature map stride 32
        # champ récepteur maximal, représentations sémantiques riches
        # ---------------------------------
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3
