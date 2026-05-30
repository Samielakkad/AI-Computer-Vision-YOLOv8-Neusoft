# -------------------------------------------------------------------------- #
#   tests/test_model.py
#   Tests légers (CPU, sans poids) qui valident que l'architecture est correcte.
#   But : prouver que le réseau se construit, que le forward pass produit les
#   bonnes formes, et que la tête découplée a le bon nombre de canaux.
#   Lancer :  pytest -q
# -------------------------------------------------------------------------- #
import torch

from nets.yolo import YoloBody

NUM_CLASSES = 20          # 20 classes VOC
INPUT = [640, 640]
# Pour une entrée 640x640 : 80*80 + 40*40 + 20*20 = 8400 ancres au total.
TOTAL_ANCHORS = 80 * 80 + 40 * 40 + 20 * 20


def _build():
    return YoloBody(INPUT, NUM_CLASSES, "s").eval()


def test_model_builds_and_forward_shapes():
    """Le réseau se construit et un forward pass renvoie les bonnes formes."""
    model = _build()
    x = torch.zeros(1, 3, *INPUT)
    with torch.no_grad():
        dbox, cls, feats, anchors, strides = model(x)

    # Trois échelles de détection (P3, P4, P5).
    assert len(feats) == 3
    # Logits de classe : (batch, num_classes, total_anchors).
    assert cls.shape == (1, NUM_CLASSES, TOTAL_ANCHORS)
    # Boîtes décodées par la DFL : (batch, 4, total_anchors).
    assert dbox.shape == (1, 4, TOTAL_ANCHORS)
    # Ancres (x, y) et strides alignées sur les prédictions.
    assert anchors.shape == (2, TOTAL_ANCHORS)
    assert strides.shape == (1, TOTAL_ANCHORS)


def test_head_channels():
    """La tête sort num_classes + reg_max*4 canaux (DFL sur 16 bins)."""
    model = _build()
    assert model.reg_max == 16
    assert model.no == NUM_CLASSES + model.reg_max * 4
    assert model.num_classes == NUM_CLASSES


def test_outputs_are_finite():
    """Aucune valeur NaN/Inf en sortie — garde-fou de stabilité numérique."""
    model = _build()
    with torch.no_grad():
        dbox, cls, *_ = model(torch.zeros(1, 3, *INPUT))
    assert torch.isfinite(dbox).all()
    assert torch.isfinite(cls).all()


if __name__ == "__main__":
    # Permet aussi de lancer `python tests/test_model.py` sans pytest.
    test_model_builds_and_forward_shapes()
    test_head_channels()
    test_outputs_are_finite()
    print("OK — tous les tests passent.")
