# -------------------------------------------------------------------------- #
#   make_diagrams.py
#   Génère les figures d'architecture du dépôt (PNG vectorisés proprement).
#   Tout est dessiné à la main avec matplotlib — aucune image importée — pour
#   que les schémas collent EXACTEMENT au code de nets/backbone.py et nets/yolo.py.
#
#   Sortie :
#     figures/yolov8_architecture.png   — backbone CSPDarknet + neck PANet + head découplé
#     figures/training_pipeline.png     — du dataset VOC au checkpoint, étape par étape
#     figures/c2f_block.png             — détail du bloc C2f (split-and-stack)
#
#   Lancer :  python figures/diagrams/make_diagrams.py
# -------------------------------------------------------------------------- #
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..")

# Palette : une couleur par grand bloc fonctionnel.
C_IN    = "#4C566A"   # entrée / sortie
C_BACK  = "#5E81AC"   # backbone
C_NECK  = "#A3BE8C"   # neck PANet
C_HEAD  = "#B48EAD"   # head découplé
C_LOSS  = "#BF616A"   # loss
C_DATA  = "#D08770"   # données
C_EDGE  = "#2E3440"
WHITE   = "#ECEFF4"


def box(ax, x, y, w, h, text, color, fs=9, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.4, edgecolor=C_EDGE, facecolor=color, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, fontweight="medium")


def arrow(ax, p1, p2, color=C_EDGE, style="-|>", lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                                 linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


# ========================================================================== #
#   FIGURE 1 — architecture complète YOLOv8
# ========================================================================== #
def fig_architecture():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8); ax.axis("off")
    ax.text(6.5, 7.65, "YOLOv8 — architecture implémentée (variante 's')",
            ha="center", fontsize=15, fontweight="bold", color=C_EDGE)
    ax.text(6.5, 7.3, "entrée 3×640×640  →  3 feature maps (strides 8/16/32)  →  boîtes + classes",
            ha="center", fontsize=9.5, color="#4C566A", style="italic")

    # ---- Backbone (colonne gauche) ----
    box(ax, 0.4, 6.2, 1.7, 0.6, "Image\n3×640×640", C_IN, 9)
    bb = [
        ("Stem  Conv 3×3 /2\n64×320×320", 5.3),
        ("Dark2  Conv+C2f\n128×160×160", 4.55),
        ("Dark3  Conv+C2f  → P3\n256×80×80", 3.8),
        ("Dark4  Conv+C2f  → P4\n512×40×40", 3.05),
        ("Dark5  Conv+C2f+SPPF → P5\n512×20×20", 2.3),
    ]
    for t, y in bb:
        box(ax, 0.4, y, 1.9, 0.62, t, C_BACK, 8)
    arrow(ax, (1.25, 6.2), (1.35, 5.92))
    for i in range(len(bb) - 1):
        arrow(ax, (1.35, bb[i][1]), (1.35, bb[i + 1][1] + 0.62))
    ax.text(1.35, 1.95, "Backbone CSPDarknet", ha="center", fontsize=9,
            fontweight="bold", color=C_BACK)

    # ---- Neck PANet (colonne centrale) ----
    nx = 3.6
    box(ax, nx, 3.8, 2.2, 0.62, "P3  256×80×80", C_NECK, 8.5)
    box(ax, nx, 3.05, 2.2, 0.62, "P4  512×40×40", C_NECK, 8.5)
    box(ax, nx, 2.3, 2.2, 0.62, "P5  512×20×20", C_NECK, 8.5)
    # liens backbone -> neck
    arrow(ax, (2.3, 4.11), (nx, 4.11), color=C_BACK, rad=0.0)
    arrow(ax, (2.3, 3.36), (nx, 3.36), color=C_BACK)
    arrow(ax, (2.3, 2.61), (nx, 2.61), color=C_BACK)
    # top-down (upsample) + bottom-up (downsample) — flèches courbes
    arrow(ax, (nx + 1.1, 2.92), (nx + 1.1, 3.67), color="#3B4252", style="-|>", rad=-0.4, lw=1.4)
    arrow(ax, (nx + 1.7, 2.92), (nx + 1.7, 3.67), color="#3B4252", style="-|>", rad=-0.4, lw=1.4)
    ax.text(nx + 1.1, 5.0, "Neck PANet\n(top-down ↑ upsample\n+ bottom-up ↓ downsample)",
            ha="center", fontsize=8.5, fontweight="bold", color="#6B8E5A")
    box(ax, nx, 5.55, 2.2, 0.95, "fusion multi-échelle\nC2f + concat", C_NECK, 8.5)
    arrow(ax, (nx + 1.1, 4.42), (nx + 1.1, 5.55), color=C_NECK, rad=0.0)

    # ---- Head découplé (colonne droite) ----
    hx = 7.4
    for (t, y, extra) in [("tête P3 / 80×80", 3.8, ""), ("tête P4 / 40×40", 3.05, ""),
                          ("tête P5 / 20×20", 2.3, "")]:
        box(ax, hx, y, 2.0, 0.62, t, C_HEAD, 8.5)
        arrow(ax, (nx + 2.2, y + 0.31), (hx, y + 0.31), color=C_NECK)
    ax.text(hx + 1.0, 1.95, "Head découplé · anchor-free", ha="center", fontsize=9,
            fontweight="bold", color=C_HEAD)

    # branches cv2 (box/DFL) et cv3 (cls)
    box(ax, 9.8, 4.35, 2.6, 0.85, "cv2 → régression boîte\nDFL (16 bins → coord. continue)", C_HEAD, 8)
    box(ax, 9.8, 3.35, 2.6, 0.7, "cv3 → classification\n20 classes (BCE)", C_HEAD, 8)
    arrow(ax, (hx + 2.0, 3.5), (9.8, 4.0), color=C_HEAD, rad=0.1)
    arrow(ax, (hx + 2.0, 3.4), (9.8, 3.6), color=C_HEAD, rad=-0.1)

    box(ax, 10.3, 2.2, 2.0, 0.7, "Décodage + NMS\n→ boîtes finales", C_IN, 8.5)
    arrow(ax, (11.1, 4.35), (11.3, 2.9), color=C_EDGE)
    arrow(ax, (11.1, 3.35), (11.3, 2.9), color=C_EDGE)

    plt.tight_layout()
    p = os.path.join(OUT, "yolov8_architecture.png")
    plt.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()
    print("écrit :", p)


# ========================================================================== #
#   FIGURE 2 — pipeline d'entraînement
# ========================================================================== #
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(6.5, 5.8, "Pipeline d'entraînement — du dataset VOC au checkpoint",
            ha="center", fontsize=15, fontweight="bold", color=C_EDGE)

    # ---- rangée principale : données -> entraînement ----
    row = 4.2
    steps = [
        ("Dataset VOC\n.jpg + .xml", C_DATA),
        ("voc_annotation.py\n2007_train/val.txt", C_DATA),
        ("DataLoader\nMosaic + MixUp", C_DATA),
        ("Phase 1 — Freeze\nbackbone gelé", C_BACK),
        ("Phase 2 — Unfreeze\nréseau complet", C_BACK),
    ]
    bw, gap, x0 = 2.0, 0.25, 0.4
    centers = []
    x = x0
    for t, c in steps:
        box(ax, x, row, bw, 0.95, t, c, 8.5)
        centers.append(x + bw / 2)
        x += bw + gap
    for i in range(len(centers) - 1):
        arrow(ax, (centers[i] + bw / 2, row + 0.47), (centers[i + 1] - bw / 2, row + 0.47))

    # ---- Loss (callout centré sous l'entraînement) ----
    box(ax, 2.6, 2.2, 5.4, 1.05,
        "Loss = 7.5·CIoU  +  0.5·BCE(cls)  +  1.5·DFL\n"
        "assignation dynamique : TaskAlignedAssigner (score^α · IoU^β)",
        C_LOSS, 9)
    # double flèche : la phase d'entraînement produit la loss, qui rétropropage
    arrow(ax, (centers[3], row), (5.3, 3.25), color=C_LOSS, rad=0.0)
    arrow(ax, (5.6, 3.25), (centers[4], row), color=C_LOSS, rad=0.0)

    # ---- EMA -> checkpoints (colonne droite, sous Phase 2) ----
    emax = centers[4] - 1.0
    box(ax, emax, 2.55, 2.0, 0.85, "EMA\npoids lissés", C_NECK, 8.5)
    arrow(ax, (centers[4], row), (emax + 1.0, 3.4), color=C_EDGE)
    box(ax, emax, 1.1, 2.0, 0.95, "Checkpoints\n+ mAP eval", C_IN, 8.5)
    arrow(ax, (emax + 1.0, 2.55), (emax + 1.0, 2.05), color=C_EDGE)

    ax.text(6.5, 0.45,
            "SGD + cosine LR (warmup) · learning rate adapté au batch size · clip-grad max_norm=10",
            ha="center", fontsize=9, color="#4C566A", style="italic")

    plt.tight_layout()
    p = os.path.join(OUT, "training_pipeline.png")
    plt.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()
    print("écrit :", p)


# ========================================================================== #
#   FIGURE 3 — détail du bloc C2f
# ========================================================================== #
def fig_c2f():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 9); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.text(4.5, 3.9, "Bloc C2f — le cœur du backbone (split-and-stack)",
            ha="center", fontsize=13, fontweight="bold", color=C_EDGE)
    ax.text(4.5, 3.55, "garde tous les états intermédiaires pour un meilleur flux de gradient",
            ha="center", fontsize=9, color="#4C566A", style="italic")

    box(ax, 0.3, 1.9, 1.3, 0.7, "entrée", C_IN, 9)
    box(ax, 2.0, 1.9, 1.5, 0.7, "Conv 1×1\nsplit → 2×c", C_BACK, 8.5)
    arrow(ax, (1.6, 2.25), (2.0, 2.25))
    # deux moitiés
    box(ax, 4.0, 2.7, 1.4, 0.6, "moitié gardée", C_NECK, 8)
    box(ax, 4.0, 1.1, 1.4, 0.6, "moitié → Bottlenecks", C_NECK, 8)
    arrow(ax, (3.5, 2.4), (4.0, 3.0), rad=0.2)
    arrow(ax, (3.5, 2.1), (4.0, 1.4), rad=-0.2)
    # bottlenecks en chaîne
    box(ax, 5.7, 1.1, 1.2, 0.6, "Bottleneck ×n\n(résiduel)", C_BACK, 7.5)
    arrow(ax, (5.4, 1.4), (5.7, 1.4))
    # concat
    box(ax, 7.1, 1.9, 1.5, 0.7, "concat\n+ Conv 1×1", C_BACK, 8.5)
    arrow(ax, (5.4, 3.0), (7.85, 2.6), rad=0.15)
    arrow(ax, (6.9, 1.4), (7.85, 1.9), rad=-0.15)
    ax.text(6.3, 2.55, "on empile chaque sortie\nintermédiaire (dense)", ha="center",
            fontsize=7.5, color="#6B8E5A")

    plt.tight_layout()
    p = os.path.join(OUT, "c2f_block.png")
    plt.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()
    print("écrit :", p)


if __name__ == "__main__":
    fig_architecture()
    fig_pipeline()
    fig_c2f()
    print("Diagrammes générés dans figures/.")
