import math
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_bbox import dist2bbox, make_anchors


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9, roll_out=False):
    """Filtre les anchor centers qui tombent à l'intérieur d'un ground truth box.

    Args:
        xy_centers (Tensor): shape(h*w, 4)  — coordonnées des centres des anchors sur la feature map
        gt_bboxes  (Tensor): shape(b, n_boxes, 4)  — boîtes GT en format xyxy
    Return:
        (Tensor): shape(b, n_boxes, h*w)  — masque booléen : 1 si l'anchor est dans la GT, 0 sinon
    """
    # Nombre total d'anchors sur la feature map (h*w)
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape

    # On vérifie que chaque anchor center est bien encadré par la GT :
    # on calcule les 4 distances (gauche, haut, droite, bas) entre le centre et
    # les bords de la GT, puis on prend le min. Si ce min > eps, l'anchor est dedans.
    if roll_out:
        # Mode roll_out : on itère sur le batch pour économiser la mémoire GPU
        # quand n_max_boxes est très grand (> roll_out_thr).
        bbox_deltas = torch.empty((bs, n_boxes, n_anchors), device=gt_bboxes.device)
        for b in range(bs):
            # gt_bboxes[b] : shape (n_boxes, 4) → on reshape en (n_boxes, 1, 4)
            # .chunk(2, 2) : découpe sur la dernière dim → lt=[x1,y1], rb=[x2,y2]
            lt, rb = gt_bboxes[b].view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
            # Pour chaque anchor : (center - lt) donne les distances aux bords gauche/haut,
            # (rb - center) donne les distances aux bords droit/bas.
            # On concatène → 4 distances, on prend le min (.amin), gt_(eps) donne le masque.
            bbox_deltas[b] = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]),
                                       dim=2).view(n_boxes, n_anchors, -1).amin(2).gt_(eps)
        return bbox_deltas
    else:
        # Mode vectorisé (batch entier d'un coup) — plus rapide quand n_boxes est raisonnable.
        # gt_bboxes.view(-1, 1, 4) → shape (b*n_boxes, 1, 4)
        # .chunk(2, 2) → lt=[x1,y1] et rb=[x2,y2], chacun shape (b*n_boxes, 1, 2)
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)

        # torch.cat sur dim=2 → on empile les 4 distances anchor↔GT
        # .view(bs, n_boxes, n_anchors, -1) → on retrouve les 4 dims
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)

        # .amin(3) : min sur les 4 distances — si positif, l'anchor est dans la GT.
        # .gt_(eps) : opération in-place, retourne un masque booléen.
        return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """Règle les conflits : si un anchor est assigné à plusieurs GT à la fois,
    on ne garde que le GT avec lequel il a le meilleur IoU.

    Pourquoi c'est nécessaire : le topk de l'assigner peut faire qu'un même
    anchor se retrouve dans le top-k de deux GT différents. Un anchor ne peut
    prédire qu'un seul objet, donc on tranche en faveur du GT le plus proche.

    Args:
        mask_pos   (Tensor): shape(b, n_max_boxes, h*w)  — masque des anchors positifs avant résolution
        overlaps   (Tensor): shape(b, n_max_boxes, h*w)  — CIoU entre chaque GT et chaque anchor
        n_max_boxes (int)  : nombre max de GT dans le batch
    Returns:
        target_gt_idx (Tensor): shape(b, h*w)              — index du GT assigné à chaque anchor
        fg_mask       (Tensor): shape(b, h*w)              — 1 si l'anchor est un foreground, 0 sinon
        mask_pos      (Tensor): shape(b, n_max_boxes, h*w) — mask_pos mis à jour (one-hot par anchor)
    """
    # b, n_max_boxes, 8400 → b, 8400
    # On somme sur l'axe GT : si la somme > 1, l'anchor est réclamé par plusieurs GT.
    fg_mask = mask_pos.sum(-2)

    # Si au moins un anchor est disputé entre plusieurs GT...
    if fg_mask.max() > 1:
        # b, n_max_boxes, 8400 — masque des anchors multi-assignés, broadcasté sur la dim GT
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])

        # b, 8400 — pour chaque anchor disputé, on récupère l'index du GT avec le meilleur IoU
        max_overlaps_idx = overlaps.argmax(1)

        # b, 8400, n_max_boxes → one-hot : seule la case du GT "gagnant" vaut 1
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)

        # b, n_max_boxes, 8400 — on remet dans le bon ordre de dimensions
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)

        # Pour les anchors disputés : on remplace mask_pos par le one-hot du gagnant.
        # Pour les anchors normaux : on garde mask_pos tel quel.
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)

        # On recalcule fg_mask après résolution des conflits
        fg_mask = mask_pos.sum(-2)

    # Pour chaque anchor foreground, l'index du GT qui lui est assigné
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    """Assigner dynamique inspiré de TOOD / PP-YOLOE.

    L'idée centrale : on ne fixe pas les anchors à l'avance comme dans l'ancien YOLOv5.
    À chaque itération, on calcule une align_metric = score^alpha * iou^beta pour
    chaque paire (anchor, GT), et on assigne les topk meilleurs anchors à chaque GT.
    Ce couplage score×IoU garantit que les anchors qu'on choisit sont à la fois
    bien localisés ET bien classifiés — d'où le nom "task-aligned".
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, roll_out_thr=0):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        # Si n_max_boxes > roll_out_thr (=64), on itère sur le batch pour économiser la VRAM
        self.roll_out_thr = roll_out_thr

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
            pd_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
            anc_points (Tensor) : shape(num_total_anchors, 2)
            gt_labels (Tensor)  : shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor)  : shape(bs, n_max_boxes, 4)
            mask_gt (Tensor)    : shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor)  : shape(bs, num_total_anchors)
            target_bboxes (Tensor)  : shape(bs, num_total_anchors, 4)
            target_scores (Tensor)  : shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor)        : shape(bs, num_total_anchors)
        """
        # Taille du batch courant
        self.bs = pd_scores.size(0)
        # Nombre max de GT dans le batch (après padding)
        self.n_max_boxes = gt_bboxes.size(1)
        # On décide si on itère sur le batch (roll_out) ou si on vectorise tout
        self.roll_out = self.n_max_boxes > self.roll_out_thr if self.roll_out_thr else False

        # Cas dégénéré : batch sans aucune GT (image entièrement background)
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        # b, max_num_obj, 8400
        # mask_pos     : anchors positifs (dans la GT + topk + mask_gt satisfait)
        # align_metric : score de la classe prédit^alpha × CIoU^beta pour chaque paire (anchor, GT)
        # overlaps     : CIoU brut entre chaque GT et chaque anchor
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)

        # target_gt_idx : b, 8400     — quel GT est assigné à chaque anchor
        # fg_mask       : b, 8400     — est-ce que cet anchor est foreground ?
        # mask_pos      : b, max_num_obj, 8400  — version one-hot après résolution des conflits
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # On récupère les targets (label, bbox, score one-hot) pour chaque anchor
        # b, 8400
        # b, 8400, 4
        # b, 8400, 80
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # On annule l'align_metric pour tous les anchors non positifs
        align_metric *= mask_pos

        # Pour chaque GT : le score d'alignement max parmi ses anchors positifs — b, max_num_obj
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)

        # Pour chaque GT : le meilleur IoU parmi ses anchors positifs — b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)

        # Normalisation : on pondère l'align_metric par le meilleur IoU, divisé par le meilleur score.
        # Ça donne une target_score normalisée entre 0 et 1 qui sert de label "soft" pour la BCE.
        # Le max sur -2 prend la valeur maximale parmi tous les GT pour chaque anchor.
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)

        # target_scores devient le label soft pour la loss de classification
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        # pd_scores : bs, num_total_anchors, num_classes
        # pd_bboxes : bs, num_total_anchors, 4
        # gt_labels : bs, n_max_boxes, 1
        # gt_bboxes : bs, n_max_boxes, 4

        # align_metric = score_classe^alpha × CIoU^beta — mesure combinée classification + localisation
        # overlaps = CIoU brut entre chaque GT et chaque anchor
        # align_metric, overlaps : bs, max_num_obj, 8400
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)

        # Un anchor est positif seulement s'il satisfait les 3 conditions simultanément :
        # 1. Son centre est physiquement à l'intérieur de la GT box
        # 2. Il fait partie des topk anchors de cette GT selon l'align_metric
        # 3. La GT existe vraiment (mask_gt = 1, pas du padding)

        # Condition 1 — b, max_num_obj, 8400
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes, roll_out=self.roll_out)

        # Condition 2 — on masque d'abord l'align_metric avec mask_in_gts pour ne
        # considérer que les anchors déjà à l'intérieur, puis on prend le topk.
        # topk_mask force à ignorer les GT sans objets (padding).
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts,
                                                topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())

        # Intersection des 3 conditions → masque final des anchors positifs
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        if self.roll_out:
            align_metric = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            overlaps = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            ind_0 = torch.empty(self.n_max_boxes, dtype=torch.long)
            for b in range(self.bs):
                ind_0[:], ind_2 = b, gt_labels[b].squeeze(-1).long()
                # On extrait le score prédit pour la bonne classe de chaque GT
                # bs, max_num_obj, 8400
                bbox_scores = pd_scores[ind_0, :, ind_2]
                # CIoU entre les GT boxes et toutes les predicted boxes
                # bs, max_num_obj, 8400
                overlaps[b] = bbox_iou(gt_bboxes[b].unsqueeze(1), pd_bboxes[b].unsqueeze(0), xywh=False,
                                       CIoU=True).squeeze(2).clamp(0)
                # align_metric = score^alpha × IoU^beta
                # alpha faible (0.5) → le score compte peu ; beta fort (6.0) → l'IoU domine.
                # C'est intentionnel : on veut des anchors bien localisés avant tout.
                align_metric[b] = bbox_scores.pow(self.alpha) * overlaps[b].pow(self.beta)
        else:
            # ind[0] : index de l'image dans le batch — shape (b, max_num_obj)
            # ind[1] : label de classe de chaque GT — shape (b, max_num_obj)
            ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)

            # Pour chaque GT, on note à quelle image du batch il appartient
            ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
            # Le label de classe de chaque GT (quelle colonne lire dans pd_scores)
            ind[1] = gt_labels.long().squeeze(-1)

            # On récupère le score prédit par le réseau pour la classe de chaque GT
            # Résultat : b, max_num_obj, 8400
            bbox_scores = pd_scores[ind[0], :, ind[1]]

            # CIoU entre toutes les GT et toutes les predicted boxes — vectorisé sur le batch entier
            # gt_bboxes.unsqueeze(2) : b, max_num_obj, 1, 4  ×  pd_bboxes.unsqueeze(1) : b, 1, 8400, 4
            # → broadcast → b, max_num_obj, 8400
            overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=True).squeeze(3).clamp(
                0)
            align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """Sélectionne les topk anchors par GT selon l'align_metric.

        Args:
            metrics     : (b, max_num_obj, h*w)  — align_metric masquée (seulement les anchors dans la GT)
            topk_mask   : (b, max_num_obj, topk) — masque pour ignorer les GT de padding
        """
        # Nombre total d'anchors sur toutes les feature maps (typiquement 8400)
        num_anchors = metrics.shape[-1]

        # b, max_num_obj, topk — indices et valeurs des topk anchors par GT
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])

        # Les GT de padding (topk_mask=False) ont leurs indices mis à 0 — ils seront ignorés
        topk_idxs[~topk_mask] = 0

        # On construit un masque is_in_topk via one-hot puis somme :
        # F.one_hot(topk_idxs, num_anchors) : b, max_num_obj, topk, 8400
        # .sum(-2) : b, max_num_obj, 8400 — 1 si l'anchor est dans le topk de ce GT
        if self.roll_out:
            is_in_topk = torch.empty(metrics.shape, dtype=torch.long, device=metrics.device)
            for b in range(len(topk_idxs)):
                is_in_topk[b] = F.one_hot(topk_idxs[b], num_anchors).sum(-2)
        else:
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)

        # Si un anchor apparaît plusieurs fois dans le topk d'un GT (cas de doublons),
        # on le remet à 0 — ça arrive rarement mais mieux vaut l'éviter.
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Construit les tenseurs de target finaux pour le calcul de loss.

        Args:
            gt_labels       : (b, max_num_obj, 1)
            gt_bboxes       : (b, max_num_obj, 4)
            target_gt_idx   : (b, h*w)   — index du GT assigné à chaque anchor
            fg_mask         : (b, h*w)   — masque foreground
        """
        # Offset de batch : pour chaque image i du batch, on décale les indices
        # de i * n_max_boxes afin d'adresser correctement le tensor flatté.
        # batch_ind : b, 1
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]

        # b, h*w — index global (dans gt_labels flatté) du GT assigné à chaque anchor
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes

        # b, h*w — label de classe pour chaque anchor (lecture dans gt_labels flatté)
        target_labels = gt_labels.long().flatten()[target_gt_idx]

        # b, h*w, 4 — boîte GT associée à chaque anchor (lecture dans gt_bboxes flatté)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        # Clip des labels (sécurité, ne devrait pas être négatif)
        target_labels.clamp(0)

        # One-hot encoding des labels → b, h*w, num_classes
        target_scores = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)

        # On met à zéro les scores des anchors background (fg_mask=0)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)


class BboxLoss(nn.Module):
    def __init__(self, reg_max=16, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # --- Loss CIoU ---
        # weight = somme des scores sur les classes → la confidence du modèle pondère la loss.
        # Plus le modèle est sûr de lui sur un anchor, plus l'erreur de localisation compte.
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)

        # CIoU entre les predicted boxes et les GT boxes, seulement sur les anchors foreground
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

        # 1 - IoU pondéré par le score, normalisé par la somme totale des scores positifs
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # --- Loss DFL (Distribution Focal Loss) ---
        if self.use_dfl:
            # On convertit les GT boxes en distances ltrb par rapport aux anchor points
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            # _df_loss calcule la loss de distribution, puis on pondère comme pour l'IoU
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391

        # L'idée du DFL : au lieu de prédire une coordonnée continue directement,
        # on prédit une distribution de probabilité sur reg_max bins discrets (0, 1, ..., reg_max-1).
        # La coordonnée finale = E[distribution] = softmax(logits) · [0,1,...,reg_max-1].
        # Ça permet au réseau d'exprimer de l'incertitude sur la position du bord.
        #
        # En pratique, une GT distance vaut souvent un nombre décimal, ex : 3.7
        # → on l'interpole entre les bins 3 (gauche) et 4 (droite).
        # tl=3, tr=4 ; wl = 4 - 3.7 = 0.3 (poids bin gauche), wr = 0.7 (poids bin droit).
        # On fait une cross_entropy vers le bin gauche et une vers le bin droit,
        # pondérées par leurs poids respectifs — c'est une interpolation de cross_entropy.
        tl = target.long()  # target left  — bin gauche (floor de la distance GT)
        tr = tl + 1         # target right — bin droit
        wl = tr - target    # weight left  — fraction du bin gauche
        wr = 1 - wl         # weight right — fraction du bin droit
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)


def xywh2xyxy(x):
    """Convertit les boîtes du format (cx, cy, w, h) vers (x1, y1, x2, y2).

    Args:
        x (np.ndarray | torch.Tensor): boîtes en format (x_centre, y_centre, largeur, hauteur)
    Returns:
        y (np.ndarray | torch.Tensor): boîtes en format (x1, y1, x2, y2) — coin supérieur gauche et inférieur droit
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


# Classe principale qui orchestre les trois composantes de la loss YOLOv8 :
# box (CIoU), cls (BCE), dfl (Distribution Focal Loss)
class Loss:
    def __init__(self, model):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.stride = model.stride  # model strides
        self.nc = model.num_classes  # number of classes
        self.no = model.no
        self.reg_max = model.reg_max

        self.use_dfl = model.reg_max > 1
        roll_out_thr = 64

        self.assigner = TaskAlignedAssigner(topk=10,
                                            num_classes=self.nc,
                                            alpha=0.5,
                                            beta=6.0,
                                            roll_out_thr=roll_out_thr)
        self.bbox_loss = BboxLoss(model.reg_max - 1, use_dfl=self.use_dfl)
        # proj : vecteur [0, 1, ..., reg_max-1] utilisé pour décoder la distribution DFL
        # E[dist] = softmax(logits) · proj
        self.proj = torch.arange(model.reg_max, dtype=torch.float)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=targets.device)
        else:
            # Index de l'image dans le batch (première colonne des targets)
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            # On crée un tensor padé à counts.max() GT par image
            out = torch.zeros(batch_size, counts.max(), 5, device=targets.device)
            # On remplit chaque slot image par image
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            # On convertit les coordonnées xywh → xyxy, puis on les remet à l'échelle de l'image originale
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            # batch, anchors, channels
            b, a, c = pred_dist.shape
            # Décodage DFL : on reshape en (b, a, 4, reg_max), on applique softmax
            # sur les reg_max bins, puis on calcule l'espérance via le produit matriciel avec proj.
            # Résultat : (b, a, 4) — les 4 distances ltrb en espace de feature map.
            # Pourquoi softmax + matmul plutôt qu'un argmax ?
            # Parce qu'on veut une valeur continue et différentiable, pas discrète.
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.to(pred_dist.device).type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        # On convertit les distances ltrb en boîtes xyxy via dist2bbox
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        # Device sur lequel tourne l'inférence
        device = preds[1].device

        # Trois composantes de la loss : [0]=box (CIoU), [1]=cls (BCE), [2]=dfl
        loss = torch.zeros(3, device=device)

        # On récupère les feature maps des trois têtes de détection
        feats = preds[2] if isinstance(preds, tuple) else preds

        # On concatène les sorties des 3 heads sur la dim spatiale (8400 = 80²+40²+20²)
        # puis on split : pred_distri (b, reg_max*4, 8400) et pred_scores (b, nc, 8400)
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # bs, num_classes + self.reg_max * 4 , 8400 =>  cls bs, num_classes, 8400;
        #                                               box bs, self.reg_max * 4, 8400
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # dtype et batch size pour les calculs suivants
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        # Taille de l'image d'entrée en pixels (h, w) — on en a besoin pour normaliser les GT
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]

        # Génère les anchor points centraux et les stride tensors pour les 8400 anchors
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # On réorganise le batch de targets :
        # colonne 0 = index image, colonne 1 = classe, colonnes 2: = coordonnées box
        targets = torch.cat((batch[:, 0].view(-1, 1), batch[:, 1].view(-1, 1), batch[:, 2:]), 1)

        # Preprocessing : padding des GT à la taille max du batch + rescaling en pixels
        # bs, max_boxes_num, 5
        targets = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])

        # bs, max_boxes_num, 5 → bs, max_boxes_num, 1 (classe) ; bs, max_boxes_num, 4 (xyxy)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy

        # mask_gt = 1 si la GT existe, 0 si c'est du padding (boîte nulle)
        # bs, max_boxes_num
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Décodage des predicted boxes : DFL → distances → boîtes xyxy
        # bs, 8400, 4
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # Assignment dynamique (TaskAligned) : pour chaque anchor, on assigne le meilleur GT
        # target_bboxes : bs, 8400, 4
        # target_scores : bs, 8400, 80   (labels soft après normalisation)
        # fg_mask       : bs, 8400       (anchors foreground)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt
        )

        # On ramène les target boxes à l'espace stride (feature map) pour être cohérent avec pred_bboxes
        target_bboxes /= stride_tensor
        # Normalisation de la loss : au moins 1 pour éviter la division par zéro
        target_scores_sum = max(target_scores.sum(), 1)

        # Loss de classification — BCE avec les target_scores soft comme labels
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Loss de localisation (CIoU + DFL) — seulement sur les anchors foreground
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        return loss.sum()  # loss(box, cls, dfl) # * batch_size


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Exponential Moving Average (EMA) des poids du modèle.

    On maintient une copie "lissée" du modèle dont les poids évoluent lentement :
        ema_weights = decay * ema_weights + (1 - decay) * model_weights
    Avec decay ≈ 0.9999, les poids EMA bougent très peu à chaque step.
    L'EMA sert à l'évaluation : les poids moyennés généralisent mieux que
    les poids bruts en fin de batch.
    Voir : https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        # La décroissance suit une rampe exponentielle au début (tau=2000 steps) :
        # decay démarre bas et monte vers 0.9999, ce qui évite que l'EMA soit
        # contaminée par les poids aléatoires des premières itérations.
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # Phase warmup : le lr monte en loi puissance depuis warmup_lr_start jusqu'au lr cible.
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
                                              ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            # Phase no-aug (fin d'entraînement) : lr fixé au minimum — on arrête l'augmentation
            # pour laisser le modèle converger proprement sur les dernières epochs.
            lr = min_lr
        else:
            # Phase principale : décroissance cosinus de lr_max vers min_lr.
            # Formule standard : lr = min_lr + 0.5*(lr_max - min_lr)*(1 + cos(pi * t/T))
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0
                    + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters - no_aug_iter)
            )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
