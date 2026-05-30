import numpy as np
import os
import torch
from torchvision.ops import nms
import pkg_resources as pkg
import config


def check_version(
    current: str = "0.0.0",
    minimum: str = "0.0.0",
    name: str = "version",
    pinned: bool = False,
) -> bool:
    current, minimum = (pkg.parse_version(str(x)) for x in (current, minimum))  # type: ignore
    result = (current == minimum) if pinned else (current >= minimum)
    return result


TORCH_2_31 = check_version(current=torch.__version__, minimum="2.3.1")


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate Anchors from Features."""
    # Listes qui vont accumuler les anchor points et leurs strides associés,
    # une entrée par niveau de feature map (P3, P4, P5).
    anchor_points, stride_tensors = [], []

    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device

    for i, stride in enumerate(strides):  # on itère sur chaque feature map et son stride
        _, _, h, w = feats[i].shape  # h, w = dimensions spatiales de la feature map courante

        sx = (
            torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        )  # shift x    # coordonnées x des centres de cellules : 0.5, 1.5, ..., w-0.5
        sy = (
            torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        )  # shift y    # idem pour l'axe y

        sy, sx = (
            torch.meshgrid(sy, sx, indexing="ij")
            if TORCH_2_31
            else torch.meshgrid(sy, sx)
        )  # meshgrid produit la grille 2D complète par produit cartésien :
        # sy[i,j] = coordonnée y de la cellule (i,j), sx[i,j] = coordonnée x.
        # Chaque cellule du grid correspond à un patch stride×stride pixels dans l'image originale.

        # On empile (sx, sy) sur la dernière dim pour obtenir [h*w, 2] — les anchor points
        # en coordonnées de feature map. view(-1, 2) aplatit la grille 2D en liste de points.
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))

        # Pour chaque anchor point, on mémorise son stride : c'est lui qui permet de
        # reprojeter les distances prédites (en unités de cellule) vers les pixels réels.
        # Forme finale : [h*w, 1]
        stride_tensors.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device)
        )
    return torch.cat(anchor_points), torch.cat(stride_tensors)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance (ltrb) to box  (xywh)"""
    """
        Le réseau YOLOv8 prédit, pour chaque anchor point, quatre distances :
        (lt_x, lt_y) = distance vers le coin supérieur-gauche du bounding box,
        (rb_x, rb_y) = distance vers le coin inférieur-droit.
        Cette fonction convertit cette représentation ltrb en coordonnées xyxy,
        puis optionnellement en xywh si le flag est activé.
    """
    # On sépare les deux paires de distances : lt = left-top, rb = right-bottom.
    # anchor_points contient les centres des cellules en coordonnées de feature map.
    # x1y1 = centre - distance_gauche/haut  →  coin supérieur-gauche
    # x2y2 = centre + distance_droite/bas   →  coin inférieur-droit
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt  # [ltx, lty]
    x2y2 = anchor_points + rb  # [rbx, rby]
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


class DecodeBox:
    def __init__(self, num_classes, input_shape):
        super(DecodeBox, self).__init__()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.bbox_attrs = 4 + num_classes

    def decode_box(self, inputs):
        # Le head YOLOv8 sort : dbox (distances ltrb DFL décodées), cls (logits classes),
        # origin_cls (brut), anchors (grille de points), strides (par niveau).
        dbox, cls, origin_cls, anchors, strides = inputs

        # dist2bbox convertit les distances ltrb → xywh en coordonnées de feature map,
        # puis on multiplie par strides pour repasser en pixels dans l'image d'entrée.
        # anchors.unsqueeze(0) ajoute la dimension batch. Forme dbox : [bs, 4, 8400]
        dbox = dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1) * strides

        # sigmoid sur les logits de classe → probabilités [0,1].
        # cat fusionne les coordonnées et les scores, permute réarrange en [bs, 8400, 4+num_classes]
        # pour que chaque ligne corresponde à une détection candidate.
        y = torch.cat((dbox, cls.sigmoid()), dim=1).permute(0, 2, 1)

        # Normalisation des coordonnées entre 0 et 1 par rapport à input_shape,
        # ordre [w, h, w, h] car dbox est en format xywh (cx, cy, w, h).
        y[:, :, :4] = y[:, :, :4] / torch.Tensor(
            [
                self.input_shape[1],
                self.input_shape[0],
                self.input_shape[1],
                self.input_shape[0],
            ]
        ).to(
            y.device()
        )  # type: ignore
        return y

    # yolo_current_boxes
    # But : transformer les bounding boxes normalisées (0~1 dans l'image d'entrée du modèle)
    # en coordonnées pixels dans l'image originale, en annulant l'effet du letterbox padding.
    # Le letterbox ajoute des bandes grises pour préserver le ratio d'aspect — si on ne
    # soustrait pas l'offset correspondant avant de rescaler, toutes les boxes sont décalées.
    # Étapes :
    # a. On swap x/y → (y, x) pour aligner avec l'ordre (height, width) de NumPy
    # b. Si letterbox : on calcule l'offset et le scale introduits par le padding,
    #    puis on les annule pour revenir dans le repère de l'image originale
    # c. On convertit xywh → xyxy en calculant coins min/max
    # d. On rescale vers les dimensions réelles de l'image source
    def yolo_current_boxes(
        self, box_xy, box_wh, input_shape, image_shape, letterbox_image
    ):
        # -----------------------------------------------------------------#
        #   On place y en premier pour faciliter la multiplication par
        #   (height, width) de l'image — NumPy travaille en ordre (row, col).
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # new_shape = dimensions de l'image originale après redimensionnement
            # proportionnel pour tenir dans input_shape (on prend le min des deux ratios
            # pour ne pas dépasser les bords).
            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (
                (input_shape - new_shape) / 2.0 / input_shape
            )  # fraction de l'image occupée par le padding de chaque côté
            scale = input_shape / new_shape  # facteur de zoom à inverser

            box_yx = (
                box_yx - offset
            ) * scale  # on retire le padding puis on "dézoom" pour revenir dans l'image originale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.0)
        box_maxes = box_yx + (box_hw / 2.0)
        boxes = np.concatenate(
            [
                box_mins[..., 0:1],
                box_mins[..., 1:2],
                box_maxes[..., 0:1],
                box_maxes[..., 1:2],
            ],
            axis=-1,
        )
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    # NMS (Non-Max Suppression)
    def non_max_suppression(
        self,
        prediction, # (batch_size, num_boxes, 5 + num_classes)
        num_classes,
        input_shape,
        image_shape,
        letterbox_image,
        conf_thres=0.5,
        nms_thres=0.4,
    ):
        box_corner = prediction.new(prediction.shape)   # copie superficielle, même shape
        # Conversion du format centre (cx, cy, w, h) → coins (x1, y1, x2, y2)
        # pour pouvoir calculer des IoU et passer à torchvision.ops.nms.
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        # Liste de résultats indexée par image dans le batch ; None = aucune détection pour l'instant.
        output = [None for _ in range(len(prediction))]

        # Pour chaque image du batch : on récupère la classe la plus probable et son score,
        # puis on filtre toutes les boxes dont le confidence score est en dessous du seuil.
        for i, image_pred in enumerate(prediction):
            class_conf, class_pred = torch.max(
                image_pred[:, 4 : 4 + num_classes], 1, keepdim=True
            )

            # On ne garde que les detections dont le score de la meilleure classe >= conf_thres.
            conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]


            if not image_pred.size(0):
                continue

            detections = torch.cat(
                (image_pred[:, :4], class_conf.float(), class_pred.float()), 1
            )

            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            # NMS par classe : c'est crucial de le faire classe par classe plutôt que
            # globalement. Deux boxes de classes différentes peuvent se superposer
            # légitimement (un chien ET une personne au même endroit) — le NMS global
            # les éliminerait à tort. En isolant chaque classe, on supprime uniquement
            # les doublons redondants pour un même objet.
            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]
                keep = nms(detections_class[:, :4], detections_class[:, 4], nms_thres)
                max_detections = detections_class[keep]

                # Premier passage sur cette image : output[i] est encore None, on l'initialise
                # directement. Passages suivants (autres classes) : on concatène.
                output = (
                    max_detections
                    if output[i] is None
                    else torch.cat((output[i], max_detections))  # type: ignore
                )

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()  # type: ignore
                # On reconstruit (cx, cy) et (w, h) depuis les coins xyxy pour pouvoir
                # appeler yolo_current_boxes qui attend ce format.
                # [:, 0:2] = x1y1 (coin supérieur-gauche), [:, 2:4] = x2y2 (coin inférieur-droit).
                box_xy, box_wh = (
                    output[i][:, 0:2] + output[i][:, 2:4] / 2,  # type: ignore
                    output[i][:, 2:4] - output[i][:, 0:2],  # type: ignore
                )

                output[i][:, :4] = self.yolo_current_boxes(  # type: ignore
                    box_xy, box_wh, input_shape, image_shape, letterbox_image
                )
                return output


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    def get_anchors_and_decode(input, input_shape, anchors, anchors_mask, num_classes):

        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        stride_h = input_shape[0] / input_height
        stride_w = input_shape[1] / input_width

        scaled_anchors = [
            (anchor_width / stride_w, anchor_height / stride_h)
            for anchor_width, anchor_height in anchors[anchors_mask[2]]
        ]

        prediction = (
            input.view(
                batch_size,
                len(anchors_mask[2]),
                num_classes + 5,
                input_height,
                input_width,
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = torch.sigmoid(prediction[..., 2])
        h = torch.sigmoid(prediction[..., 3])
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        grid_x = (
            torch.linspace(0, input_width - 1, input_width)
            .repeat(input_height, 1)
            .repeat(batch_size * len(anchors_mask[2]), 1, 1)
            .view(x.shape)
            .type(FloatTensor)
        )  # type: ignore
        grid_y = (
            torch.linspace(0, input_height - 1, input_height)
            .repeat(input_width, 1)
            .t()
            .repeat(batch_size * len(anchors_mask[2]), 1, 1)
            .view(y.shape)
            .type(FloatTensor)
        )  # type: ignore

        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = (
            anchor_w.repeat(batch_size, 1)
            .repeat(1, 1, input_height * input_width)
            .view(w.shape)
        )
        anchor_h = (
            anchor_h.repeat(batch_size, 1)
            .repeat(1, 1, input_height * input_width)
            .view(h.shape)
        )

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data * 2.0 - 0.5 + grid_x
        pred_boxes[..., 1] = y.data * 2.0 - 0.5 + grid_y
        pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
        pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

        point_h = 5
        point_w = 5

        box_xy = pred_boxes[..., 0:2].cpu().numpy() * 32

        box_wh = pred_boxes[..., 2:4].cpu().numpy() * 32

        # Conversion des coordonnées de grid en pixels (stride = 32 ici) pour l'affichage
        grid_x = grid_x.cpu().numpy() * 32
        grid_y = grid_y.cpu().numpy() * 32

        # Idem pour les dimensions des anchors : on repasse en pixels
        anchor_w = anchor_w.cpu().numpy() * 32
        anchor_h = anchor_h.cpu().numpy() * 32

        fig = plt.figure()
        # Premier sous-plot : anchor boxes superposées à l'image
        ax = fig.add_subplot(121)
        # Sous-plot 1/2 — visualisation des anchor boxes sur l'image originale
        img = Image.open(os.path.join(config.vocPath, "JPEGImages/000003.jpg")).resize(
            [640, 640]
        )
        # Affichage de l'image avec transparence pour voir les overlays dessus
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)

        # Marges élargies pour visualiser les boxes qui débordent légèrement du cadre image
        plt.scatter(grid_x, grid_y)
        # On marque la cellule d'intérêt (point_h, point_w) en noir pour la distinguer
        plt.scatter(point_h * 32, point_w * 32, c="black")
        # Inversion de l'axe y : en image, y=0 est en haut, matplotlib met y=0 en bas par défaut
        plt.gca().invert_yaxis()

        # Calcul du coin supérieur-gauche de chaque anchor pour tracer les rectangles
        anchor_left = grid_x - anchor_w / 2
        anchor_top = grid_y - anchor_h / 2

        # Trois anchors de tailles différentes centrés sur la cellule (point_h, point_w) —
        # rect1, rect2, rect3 correspondent aux trois ratios de la tête de détection courante.
        # Rectangle(coin_bas_gauche, largeur, hauteur) en coordonnées matplotlib.
        rect1 = plt.Rectangle(
            [anchor_left[0, 0, point_h, point_w], anchor_top[0, 0, point_h, point_w]],
            anchor_w[0, 0, point_h, point_w],
            anchor_h[0, 0, point_h, point_w],
            color="r",
            fill=False,
        )
        rect2 = plt.Rectangle(
            [anchor_left[0, 1, point_h, point_w], anchor_top[0, 1, point_h, point_w]],
            anchor_w[0, 1, point_h, point_w],
            anchor_h[0, 1, point_h, point_w],
            color="r",
            fill=False,
        )
        rect3 = plt.Rectangle(
            [anchor_left[0, 2, point_h, point_w], anchor_top[0, 2, point_h, point_w]],
            anchor_w[0, 2, point_h, point_w],
            anchor_h[0, 2, point_h, point_w],
            color="r",
            fill=False,
        )

        # Ajout des rectangles anchor au sous-plot
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        # Deuxième sous-plot : bounding boxes prédites (après décodage) vs anchors
        ax = fig.add_subplot(122)
        # Même image, même transparence — on compare visuellement anchors vs prédictions
        plt.imshow(img, alpha=0.5)
        plt.ylim(-30, 650)
        plt.xlim(-30, 650)

        # Grid et cellule cible identiques au premier subplot pour faciliter la comparaison
        plt.scatter(grid_x, grid_y)
        plt.scatter(point_h * 32, point_w * 32, c="black")
        # box_xy contient les centres des bounding boxes décodées — on les affiche en rouge
        plt.scatter(
            box_xy[0, :, point_h, point_w, 0], box_xy[0, :, point_h, point_w, 1], c="r"
        )
        plt.gca().invert_yaxis()

        pre_left = box_xy[..., 0] - box_wh[..., 0] / 2
        pre_top = box_xy[..., 1] - box_wh[..., 1] / 2

        # Tracé des bounding boxes prédites pour la cellule (point_h, point_w).
        # pre_left/pre_top = coin supérieur-gauche, box_wh = largeur/hauteur décodées.
        rect1 = plt.Rectangle(
            [pre_left[0, 0, point_h, point_w], pre_top[0, 0, point_h, point_w]],
            box_wh[0, 0, point_h, point_w, 0],
            box_wh[0, 0, point_h, point_w, 1],
            color="r",
            fill=False,
        )
        rect2 = plt.Rectangle(
            [pre_left[0, 1, point_h, point_w], pre_top[0, 1, point_h, point_w]],
            box_wh[0, 1, point_h, point_w, 0],
            box_wh[0, 1, point_h, point_w, 1],
            color="r",
            fill=False,
        )
        rect3 = plt.Rectangle(
            [pre_left[0, 2, point_h, point_w], pre_top[0, 2, point_h, point_w]],
            box_wh[0, 2, point_h, point_w, 0],
            box_wh[0, 2, point_h, point_w, 1],
            color="r",
            fill=False,
        )

        # Ajout des bounding boxes prédites au sous-plot
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        # Affichage final des deux sous-plots côte à côte
        plt.show()

    # batch_size = 4 images ; 255 canaux = 3 anchors × (5 + 80 classes COCO) ;
    # 20×20 = taille de la feature map pour le niveau de détection des grands objets (stride 32)

    feat = torch.from_numpy(np.random.normal(0.2, 0.5, [4, 255, 20, 20])).float()

    anchors = np.array(
        [
            [116, 90],
            [156, 198],
            [373, 326],
            [30, 61],
            [62, 45],
            [59, 119],
            [10, 13],
            [16, 30],
            [33, 23],
        ]
    )

    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    get_anchors_and_decode(feat, [640, 640], anchors, anchors_mask, 80)
