from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(
        self,
        annotation_lines,
        input_shape,
        num_classes,
        epoch_length,
        mosaic,
        mixup,
        mosaic_prob,
        mixup_prob,
        train,
        special_aug_ratio=0.7,
    ):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio

        self.epoch_now = -1
        self.length = len(self.annotation_lines)

        self.bbox_attrs = 5 + num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        # ---------------------------------------------------#
        #   En entraînement on applique les augmentations aléatoires.
        #   En validation on coupe tout ça — pas question de modifier
        #   les images sur lesquelles on mesure la vraie performance.
        # ---------------------------------------------------#
        if (
            self.mosaic
            and self.rand() < self.mosaic_prob
            and self.epoch_now < self.epoch_length * self.special_aug_ratio
        ):
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)

            if self.mixup and self.rand() < self.mixup_prob:
                lines = sample(self.annotation_lines, 1)
                image_2, box_2 = self.get_random_data(
                    lines[0], self.input_shape, random=self.train
                )
                image, box = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            image, box = self.get_random_data(
                self.annotation_lines[index], self.input_shape, random=self.train
            )

        image = np.transpose(
            preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1)
        )
        box = np.array(box, dtype=np.float32)

        # ---------------------------------------------------#
        #   Mise en forme des ground truth boxes pour l'entraînement.
        #   On pré-alloue labels_out en (nL, 6) : la colonne 0 servira
        #   d'index de batch dans le collate, les colonnes 1-5 portent
        #   class + cx + cy + w + h normalisés.
        # ---------------------------------------------------#
        nL = len(box)
        labels_out = np.zeros((nL, 6))
        if nL:
            # ---------------------------------------------------#
            #   Normalisation des coordonnées dans [0, 1].
            #   x1/x2 divisés par la largeur, y1/y2 par la hauteur —
            #   indispensable pour que le réseau soit indépendant
            #   de la résolution absolue d'entrée.
            # ---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            # ---------------------------------------------------#
            #   Conversion format coin-à-coin → format YOLO (cx, cy, w, h).
            #   Colonnes 0-1 : coin supérieur-gauche (x1, y1)  →  centre (cx, cy)
            #   Colonnes 2-3 : coin inférieur-droit  (x2, y2)  →  dimensions (w, h)
            #   Colonne  4   : indice de classe
            # ---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

            # ---------------------------------------------------#
            #   Remplissage de labels_out dans l'ordre attendu par la loss.
            #   La colonne 0 (index batch) est remplie dans le collate,
            #   pas ici — on la laisse à 0 pour l'instant.
            # ---------------------------------------------------#
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]

        return image, labels_out

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(
        self,
        annotation_line,
        input_shape,
        jitter=0.3,
        hue=0.1,
        sat=0.7,
        val=0.4,
        random=True,
    ):
        line = annotation_line.split()
        # ------------------------------#
        #   Lecture de l'image et conversion en RGB.
        #   cvtColor gère les PNG RGBA et les JPEG en niveaux de gris
        #   pour qu'on ait toujours 3 canaux en entrée.
        # ------------------------------#
        image = Image.open(line[0])
        image = cvtColor(image)
        # ------------------------------#
        #   Dimensions originales de l'image et dimensions cibles.
        #   iw/ih = taille source, w/h = taille réseau (ex. 640×640).
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape
        # ------------------------------#
        #   Parsing des ground truth boxes depuis la ligne d'annotation.
        #   Format attendu : "x1,y1,x2,y2,class" par box, en pixels absolus.
        # ------------------------------#
        box = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   Letterbox : on redimensionne en gardant le ratio,
            #   puis on complète les bords avec du gris (128, 128, 128).
            #   Pourquoi gris ? C'est la valeur neutre après normalisation,
            #   ça évite de biaiser les activations sur les zones de padding.
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # ---------------------------------#
            #   Les boxes doivent suivre la même transformation affine
            #   que l'image. On recale puis on clippe pour éliminer
            #   tout ce qui déborde hors du canvas final.
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box

        # ------------------------------------------#
        #   Distorsion d'aspect ratio + changement d'échelle aléatoires.
        #   Le jitter sur new_ar simule des images étirées/compressées
        #   — ça force le réseau à ne pas se fier aux proportions exactes
        #   des objets pour les reconnaître.
        # ------------------------------------------#
        new_ar = (
            iw
            / ih
            * self.rand(1 - jitter, 1 + jitter)
            / self.rand(1 - jitter, 1 + jitter)
        )
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   Placement aléatoire de l'image redimensionnée dans le canvas.
        #   Les bords non couverts reçoivent le gris neutre (128).
        #   dx/dy aléatoires = translation implicite, autre forme
        #   d'augmentation de position.
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ------------------------------------------#
        #   Flip horizontal aléatoire (p = 0.5).
        #   Simple mais très efficace — double la diversité de vues
        #   sans aucun coût d'annotation supplémentaire.
        # ------------------------------------------#
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   Jitter HSV : on perturbe teinte, saturation et luminosité
        #   de façon indépendante via des LUT 256 valeurs.
        #   Pourquoi HSV et pas RGB ? Parce que HSV sépare la couleur
        #   (H) de l'intensité (V), ce qui permet de simuler des
        #   conditions d'éclairage réalistes sans saturer les canaux.
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   Conversion RGB → HSV pour appliquer les perturbations
        #   canal par canal avec les LUT calculées ci-dessous.
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   Construction des LUT d'intensité (Look-Up Tables).
        #   lut_hue : rotation circulaire modulo 180° (H va de 0 à 179 en OpenCV).
        #   lut_sat, lut_val : scaling clippé dans [0, 255].
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
        )
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   Recalage des ground truth boxes après toutes les transformations
        #   géométriques (scale, translation, flip).
        #   Clippage obligatoire : le scaling + translation peuvent pousser
        #   des coins hors du canvas — une box invalide ferait crasher la loss.
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(
        self, annotation_line, input_shape, jitter=0.3, hue=0.1, sat=0.7, val=0.4
    ):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:
            # ---------------------------------#
            #   Parsing de la ligne d'annotation pour récupérer
            #   le chemin de l'image et les boxes associées.
            # ---------------------------------#
            line_content = line.split()
            # ---------------------------------#
            #   Ouverture de l'image et conversion RGB.
            # ---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)

            # ---------------------------------#
            #   Dimensions originales de cette image du mosaic.
            # ---------------------------------#
            iw, ih = image.size
            # ---------------------------------#
            #   Boxes en pixels absolus pour cette image.
            # ---------------------------------#
            box = np.array(
                [np.array(list(map(int, box.split(",")))) for box in line_content[1:]]
            )

            # ---------------------------------#
            #   Flip horizontal indépendant pour chaque image du mosaic.
            #   Chacune des 4 tuiles peut être retournée ou non,
            #   ce qui multiplie la diversité de combinaisons.
            # ---------------------------------#
            flip = self.rand() < 0.5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # ------------------------------------------#
            #   Distorsion d'aspect ratio + scaling pour cette tuile.
            #   Le mosaic combine 4 images à des échelles différentes —
            #   c'est ce qui force le réseau à détecter des petits objets :
            #   une image zoomée à 0.4× produit des objets minuscules
            #   dans la tuile finale.
            # ------------------------------------------#
            new_ar = (
                iw
                / ih
                * self.rand(1 - jitter, 1 + jitter)
                / self.rand(1 - jitter, 1 + jitter)
            )
            scale = self.rand(0.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # -----------------------------------------------#
            #   Positionnement des 4 tuiles autour du point de coupe (cutx, cuty).
            #   index 0 : haut-gauche   (collée contre le coin de coupe)
            #   index 1 : bas-gauche    (décalée vers le bas)
            #   index 2 : bas-droit     (décalée bas + droite)
            #   index 3 : haut-droit    (décalée vers la droite)
            #   dx/dy négatifs = la tuile dépasse vers le haut/gauche,
            #   le canvas (gris) masque ce qui sort des bords.
            # -----------------------------------------------#
            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            # ---------------------------------#
            #   Recalage des boxes dans le repère du canvas final.
            #   Même logique que dans get_random_data : scale + translation
            #   puis clippage strict — une box à moitié hors canvas reste
            #   utilisable, une box entièrement dehors (w ou h ≤ 1) est
            #   supprimée pour éviter des gradients parasites.
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[: len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # ---------------------------------#
        #   Assemblage final des 4 tuiles en une seule image mosaic.
        #   On découpe chaque image_data exactement sur son quadrant —
        #   le résultat est une image [h, w, 3] avec 4 scènes distinctes.
        #   Intérêt principal : chaque batch voit des contextes visuels
        #   variés et des objets à des échelles très différentes, ce qui
        #   booste la détection de petits objets bien mieux qu'un simple
        #   random crop.
        # ---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)
        # ---------------------------------#
        #   Jitter HSV appliqué sur l'image mosaic complète après assemblage.
        #   On recalcule les facteurs r une fois pour toute l'image — les 4 tuiles
        #   subissent ainsi la même perturbation colorimétrique, ce qui évite des
        #   transitions de teinte visibles aux joints de coupe.
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   Conversion RGB → HSV pour travailler séparément sur
        #   teinte, saturation et luminosité.
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        # ---------------------------------#
        #   LUT identiques à get_random_data — voir commentaires là-bas.
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
        )
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   Fusion des boxes des 4 tuiles : on ne garde que les boxes
        #   visibles dans leur quadrant respectif (merge_bboxes clippe
        #   celles qui débordent sur le mauvais côté de la ligne de coupe).
        # ---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = (
            np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        )
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes


# Fonction collate utilisée par le DataLoader pour assembler un batch.
# PyTorch appelle cette fonction sur la liste de (image, labels_out) renvoyés
# par __getitem__. On y estampille chaque label avec son index dans le batch
# (colonne 0) avant de tout concaténer — c'est ce qui permet à la loss de savoir
# à quelle image du batch appartient chaque ground truth box.
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)

    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes


# # Version alternative du collate avec padding à la taille maximale de boxes.
# # Utile si on veut un tensor (bs, n_max_boxes, 4) de taille fixe au lieu
# # d'une concaténation — certaines architectures l'exigent, pas YOLOv8 standard.
# def yolo_dataset_collate(batch):
#     images      = []
#     n_max_boxes = 0
#     bs          = len(batch)
#     for i, (img, box) in enumerate(batch):
#         images.append(img)
#         n_max_boxes = max(n_max_boxes, len(box))

#     bboxes  = torch.zeros((bs, n_max_boxes, 4))
#     labels  = torch.zeros((bs, n_max_boxes, 1))
#     masks   = torch.zeros((bs, n_max_boxes, 1))

#     for i, (img, box) in enumerate(batch):
#         _sub_length = len(box)
#         bboxes[i, :_sub_length] = box[:, :4]
#         labels[i, :_sub_length] = box[:, 4]
#         masks[i, :_sub_length]  = 1

#     images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
#     bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
#     return images, bboxes, labels, masks
