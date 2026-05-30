# -------------------------------------------------------------------------- #
#   voc_annotation.py
#   Prépare les fichiers que l'entraînement va réellement lire.
# -------------------------------------------------------------------------- #
#   Deux choses se passent ici :
#     1. On découpe le dataset en train / val / test et on écrit les listes
#        d'identifiants dans VOCdevkit/VOC2007/ImageSets/Main/.
#     2. On parcourt les .xml VOC pour produire 2007_train.txt et 2007_val.txt :
#        une ligne = chemin de l'image suivie de ses boîtes "xmin,ymin,xmax,ymax,classe".
#        C'est exactement le format que le DataLoader attend.
#
#   À lancer UNE fois avant le premier entraînement (et à relancer si on change
#   le dataset ou la liste de classes). Tous les chemins viennent de config.py,
#   donc rien n'est codé en dur.
# -------------------------------------------------------------------------- #
import os
import random
import xml.etree.ElementTree as ET

import numpy as np

import config
from utils.utils import get_classes

# annotation_mode pilote ce que le script calcule :
#   0 -> tout le pipeline : ImageSets + 2007_train.txt / 2007_val.txt
#   1 -> seulement les listes ImageSets/Main/*.txt
#   2 -> seulement 2007_train.txt / 2007_val.txt (suppose les ImageSets déjà faits)
annotation_mode = 0

# Liste des classes — DOIT être la même que celle utilisée à l'entraînement.
# Si 2007_train.txt sort vide, c'est presque toujours que les noms de classes
# ici ne correspondent pas aux <name> des .xml.
classes_path = config.vocClassesPath

# Découpe du dataset.
#   trainval_percent : part de (train + val) vs test            -> ici 90 % / 10 %
#   train_percent    : part de train à l'intérieur de trainval  -> ici 90 % / 10 %
# Donc au final : ~81 % train, ~9 % val, ~10 % test.
trainval_percent = 0.9
train_percent = 0.9

# Racine du dataset VOC (VOCdevkit/), récupérée depuis config.py.
VOCdevkit_path = config.vocDevkitPath

# Quels sous-ensembles transformer en fichiers d'annotation finaux.
VOCdevkit_sets = [("2007", "train"), ("2007", "val")]
classes, _ = get_classes(classes_path)

# Compteurs de stats, juste pour afficher un récap à la fin.
photo_nums = np.zeros(len(VOCdevkit_sets))
nums = np.zeros(len(classes))


def convert_annotation(year, image_id, list_file):
    """Lit un .xml VOC et ajoute ses boîtes à la ligne courante de list_file.

    Pour chaque <object> de l'image : on saute les objets marqués `difficult`
    ou dont la classe n'est pas dans notre liste, puis on écrit la boîte au
    format "xmin,ymin,xmax,ymax,id_classe" collée au chemin de l'image.
    """
    in_file = open(
        os.path.join(VOCdevkit_path, "VOC%s/Annotations/%s.xml" % (year, image_id)),
        encoding="utf-8",
    )
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter("object"):
        difficult = 0
        if obj.find("difficult") is not None:
            difficult = obj.find("difficult").text
        cls = obj.find("name").text
        # On ignore les classes hors-liste et les objets "difficiles"
        # (occlus, ambigus) — ils bruitent l'apprentissage.
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")
        b = (
            int(float(xmlbox.find("xmin").text)),
            int(float(xmlbox.find("ymin").text)),
            int(float(xmlbox.find("xmax").text)),
            int(float(xmlbox.find("ymax").text)),
        )
        list_file.write(" " + ",".join([str(a) for a in b]) + "," + str(cls_id))
        nums[cls_id] += 1


def print_table(columns, widths):
    """Petit affichage en colonnes pour le récap classe -> nb d'objets."""
    for i in range(len(columns[0])):
        print("|", end=" ")
        for j in range(len(columns)):
            print(columns[j][i].rjust(int(widths[j])), end=" ")
            print("|", end=" ")
        print()


if __name__ == "__main__":
    # Seed fixe : le découpage train/val/test doit être reproductible d'un run
    # à l'autre, sinon on ne compare plus rien.
    random.seed(0)

    # Le pipeline VOC ne tolère pas les espaces dans le chemin du dataset.
    if " " in os.path.abspath(VOCdevkit_path):
        raise ValueError(
            "Le chemin du dataset (et les noms d'images) ne doit contenir aucun espace."
        )

    # ---------------------------------------------------------------------- #
    #   Étape 1 — générer les listes d'identifiants dans ImageSets/Main/
    # ---------------------------------------------------------------------- #
    if annotation_mode in (0, 1):
        print("Génération des .txt dans ImageSets/Main ...")
        xmlfilePath = os.path.join(VOCdevkit_path, "VOC2007/Annotations")
        saveBasePath = os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main")

        # On ne garde que les vrais .xml pour construire la liste des images.
        total_xml = [x for x in os.listdir(xmlfilePath) if x.endswith(".xml")]
        num = len(total_xml)
        indices = range(num)

        # Tirage aléatoire des deux découpes imbriquées.
        tv = int(num * trainval_percent)   # taille de trainval
        tr = int(tv * train_percent)       # taille de train dans trainval
        trainval = random.sample(indices, tv)
        train = random.sample(trainval, tr)

        print("taille trainval :", tv)
        print("taille train    :", tr)

        ftrainval = open(os.path.join(saveBasePath, "trainval.txt"), "w")
        ftest = open(os.path.join(saveBasePath, "test.txt"), "w")
        ftrain = open(os.path.join(saveBasePath, "train.txt"), "w")
        fval = open(os.path.join(saveBasePath, "val.txt"), "w")

        for i in indices:
            name = total_xml[i][:-4] + "\n"
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

        # On referme proprement chaque flux (sinon risque d'écriture tronquée).
        ftrainval.close()
        ftrain.close()
        fval.close()
        ftest.close()
        print("ImageSets/Main : OK.")

    # ---------------------------------------------------------------------- #
    #   Étape 2 — écrire 2007_train.txt et 2007_val.txt à partir des .xml
    # ---------------------------------------------------------------------- #
    if annotation_mode in (0, 2):
        print("Génération de 2007_train.txt et 2007_val.txt ...")
        type_index = 0
        for year, image_set in VOCdevkit_sets:
            image_ids = (
                open(
                    os.path.join(
                        VOCdevkit_path, "VOC%s/ImageSets/Main/%s.txt" % (year, image_set)
                    ),
                    encoding="utf-8",
                )
                .read()
                .strip()
                .split()
            )
            list_file = open("%s_%s.txt" % (year, image_set), "w", encoding="utf-8")
            for image_id in image_ids:
                # Chemin absolu de l'image, puis ses boîtes sur la même ligne.
                list_file.write(
                    "%s/VOC%s/JPEGImages/%s.jpg"
                    % (os.path.abspath(VOCdevkit_path), year, image_id)
                )
                convert_annotation(year, image_id, list_file)
                list_file.write("\n")  # une image par ligne
            photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("2007_train.txt et 2007_val.txt : OK.")

        # ------------------------------------------------------------------ #
        #   Récap : nombre d'objets par classe + garde-fous
        # ------------------------------------------------------------------ #
        str_nums = [str(int(x)) for x in nums]
        table = [classes, str_nums]
        widths = [max(len(s) for s in col) for col in table]
        print_table(table, widths)

        if photo_nums[0] <= 500:
            print(
                "Moins de 500 images d'entraînement : dataset petit, penser à augmenter "
                "le nombre d'epochs pour avoir assez de steps de descente de gradient."
            )
        if np.sum(nums) == 0:
            print(
                "Aucun objet trouvé dans le dataset : vérifier classes_path et que les "
                "noms de classes correspondent bien aux <name> des fichiers .xml, "
                "sinon l'entraînement n'apprendra rien."
            )
