# -------------------------------------------------------------------------- #
#   config.py — point unique de vérité pour tous les chemins du projet
# -------------------------------------------------------------------------- #
#   J'en ai eu marre de retrouver des chemins absolus "F:\..." codés en dur un
#   peu partout : ça casse dès qu'on change de machine. Ici tout est résolu
#   relativement à l'emplacement de CE fichier, donc le dépôt tourne tel quel
#   après un simple `git clone`, peu importe où il est posé.
#
#   Tout le reste du code (train.py, voc_annotation.py, ...) importe ces
#   variables au lieu de réécrire des chemins à la main.
# -------------------------------------------------------------------------- #
import os

# Racine du projet = dossier qui contient ce fichier config.py.
ROOT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------ Jeu de données ----------------------------- #
#   Dataset au format VOC : VOCdevkit/VOC2007/{JPEGImages, Annotations, ImageSets}
vocDevkitPath = os.path.join(ROOT, "VOCdevkit")
vocPath       = os.path.join(vocDevkitPath, "VOC2007")

# Liste des 20 classes VOC, une par ligne (person, car, dog, ...).
vocClassesPath = os.path.join(ROOT, "model_data", "voc_classes.txt")

# ----------------------- Fichiers d'annotation générés --------------------- #
#   Produits par voc_annotation.py : une ligne = chemin image + boîtes
#   "xmin,ymin,xmax,ymax,classe". C'est ce que le DataLoader lit pendant
#   l'entraînement.
train_annotation_path = os.path.join(ROOT, "2007_train.txt")
val_annotation_path   = os.path.join(ROOT, "2007_val.txt")

# ----------------------------- Poids pré-entraînés ------------------------- #
#   Backbone YOLOv8-s pré-entraîné : on part TOUJOURS de là, jamais de zéro.
#   Sans ces poids l'extraction de features est aléatoire et le réseau
#   n'apprend quasiment rien sur un dataset de cette taille.
pretrained_pth_Path = os.path.join(ROOT, "model_data", "yolov8_s.pth")
