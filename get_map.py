# -------------------------------------------------------------------------- #
#   get_map.py — évaluation de la précision du détecteur (mAP)
# -------------------------------------------------------------------------- #
#   Recall et Precision ne sont pas des valeurs fixes comme l'AP : elles
#   dépendent du seuil de confidence choisi. Pour tracer correctement la courbe
#   precision/recall et calculer le mAP, on doit donc récupérer QUASIMENT toutes
#   les prédictions (confidence très basse, ~0.001) — c'est pour ça que le
#   detection-results/ ici contient bien plus de boîtes qu'un predict normal.
#   Le mAP se calcule ensuite en confrontant ces prédictions à la vérité terrain
#   extraite des .xml VOC.
# -------------------------------------------------------------------------- #
import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

if __name__ == "__main__":
    # ---------------------------------------------------------------------- #
    #   map_mode pilote ce que ce fichier calcule :
    #     0 -> pipeline complet (prédictions + vérité terrain + calcul du mAP)
    #     1 -> uniquement les prédictions
    #     2 -> uniquement la vérité terrain
    #     3 -> uniquement le calcul du VOC mAP (suppose 1 et 2 déjà faits)
    #     4 -> mAP COCO 0.50:0.95 via pycocotools (nécessite 1 et 2 + pycocotools)
    # ---------------------------------------------------------------------- #
    map_mode = 0

    # -------------------------------------------------------#
    #   Classes sur lesquelles mesurer le mAP — mêmes que l'entraînement.
    # -------------------------------------------------------#
    classes_path = "model_data/voc_classes.txt"

    # -------------------------------------------------------#
    #   MINOVERLAP : seuil d'IoU pour qu'une prédiction compte comme correcte.
    #   0.5 -> on calcule le mAP@0.5 (la métrique VOC classique).
    #   Mettre 0.75 donnerait un mAP@0.75, plus exigeant sur la localisation.
    # -------------------------------------------------------#
    MINOVERLAP = 0.5

    # -------------------------------------------------------#
    #   confidence très bas exprès : on veut presque toutes les boîtes pour
    #   balayer tous les seuils possibles de recall/precision.
    # -------------------------------------------------------#
    confidence = 0.001
    # -------------------------------------------------------#
    #   nms_iou : NMS volontairement peu agressif ici (0.5) pour ne pas
    #   supprimer trop tôt des boîtes utiles au calcul du mAP.
    # -------------------------------------------------------#
    nms_iou = 0.5

    # -------------------------------------------------------#
    #   score_threhold sert seulement à afficher Recall/Precision à un seuil
    #   de référence — il n'influence pas la valeur du mAP lui-même.
    # -------------------------------------------------------#
    score_threhold = 0.5

    # -------------------------------------------------------#
    #   map_vis : sauvegarder ou non les images annotées pendant l'évaluation.
    # -------------------------------------------------------#
    map_vis = False

    # Racine du dataset VOC et dossier de sortie des résultats intermédiaires.
    VOCdevkit_path = "VOCdevkit"
    map_out_path = "map_out"

    # On évalue sur l'ensemble de test listé dans ImageSets/Main/test.txt.
    image_ids = (
        open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt"))
        .read()
        .strip()
        .split()
    )

    # Arborescence de sortie : prédictions, vérité terrain, images optionnelles.
    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, "ground-truth")):
        os.makedirs(os.path.join(map_out_path, "ground-truth"))
    if not os.path.exists(os.path.join(map_out_path, "detection-results")):
        os.makedirs(os.path.join(map_out_path, "detection-results"))
    if not os.path.exists(os.path.join(map_out_path, "images-optional")):
        os.makedirs(os.path.join(map_out_path, "images-optional"))

    class_names, _ = get_classes(classes_path)

    # ---------------------------------------------------------------------- #
    #   Étape 1 — prédictions du modèle sur chaque image de test
    # ---------------------------------------------------------------------- #
    if map_mode == 0 or map_mode == 1:
        print("Chargement du modèle.")
        # confidence/nms_iou bas forcés ici pour l'évaluation (voir plus haut).
        yolo = YOLO(confidence=confidence, nms_iou=nms_iou)
        print("Modèle chargé.")
        print("Calcul des prédictions.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(
                VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg"
            )
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            # Écrit detection-results/<image_id>.txt : "classe score left top right bottom".
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Prédictions terminées.")

    # ---------------------------------------------------------------------- #
    #   Étape 2 — vérité terrain extraite des annotations VOC
    # ---------------------------------------------------------------------- #
    if map_mode == 0 or map_mode == 2:
        print("Extraction de la vérité terrain.")
        for image_id in tqdm(image_ids):
            with open(
                os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w"
            ) as new_f:
                root = ET.parse(
                    os.path.join(
                        VOCdevkit_path, "VOC2007/Annotations/" + image_id + ".xml"
                    )
                ).getroot()
                for obj in root.findall("object"):
                    difficult_flag = False
                    if obj.find("difficult") is not None:
                        difficult = obj.find("difficult").text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find("name").text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find("bndbox")
                    left = bndbox.find("xmin").text
                    top = bndbox.find("ymin").text
                    right = bndbox.find("xmax").text
                    bottom = bndbox.find("ymax").text

                    # Les objets "difficult" sont marqués : la métrique VOC les
                    # ignore au lieu de les compter comme faux négatifs.
                    if difficult_flag:
                        new_f.write(
                            "%s %s %s %s %s difficult\n"
                            % (obj_name, left, top, right, bottom)
                        )
                    else:
                        new_f.write(
                            "%s %s %s %s %s\n" % (obj_name, left, top, right, bottom)
                        )
        print("Vérité terrain terminée.")

    # ---------------------------------------------------------------------- #
    #   Étape 3 — calcul du VOC mAP à partir des deux dossiers ci-dessus
    # ---------------------------------------------------------------------- #
    if map_mode == 0 or map_mode == 3:
        print("Calcul du mAP.")
        get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
        print("mAP calculé.")

    # ---------------------------------------------------------------------- #
    #   Variante COCO — mAP moyenné sur les IoU 0.50:0.95 (pycocotools requis)
    # ---------------------------------------------------------------------- #
    if map_mode == 4:
        print("Calcul du mAP COCO.")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("mAP COCO calculé.")
