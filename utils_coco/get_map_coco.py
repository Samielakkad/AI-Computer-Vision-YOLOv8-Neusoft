import json
import os

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from utils.utils import cvtColor, preprocess_input, resize_image
from yolo import YOLO

#---------------------------------------------------------------------------#
#   map_mode détermine ce qui est calculé à l'exécution :
#   map_mode = 0 : pipeline complet — inférence + calcul du mAP.
#   map_mode = 1 : inférence uniquement (génère les prédictions).
#   map_mode = 2 : calcul du mAP uniquement (à partir des prédictions existantes).
#---------------------------------------------------------------------------#
map_mode            = 0
#-------------------------------------------------------#
#   Chemin vers les annotations et les images du jeu de validation COCO.
#-------------------------------------------------------#
cocoGt_path         = 'coco_dataset/annotations/instances_val2017.json'
dataset_img_path    = 'coco_dataset/val2017'
#-------------------------------------------------------#
#   Dossier de sortie pour les résultats temporaires (par défaut : map_out).
#-------------------------------------------------------#
temp_save_path      = 'map_out/coco_eval'

class mAP_YOLO(YOLO):
    #---------------------------------------------------#
    #   Inférence sur une image et collecte des résultats.
    #---------------------------------------------------#
    def detect_image(self, image_id, image, results, clsid2catid):
        #---------------------------------------------------#
        #   Récupération des dimensions (hauteur, largeur) de l'image d'entrée.
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Conversion en RGB pour éviter les erreurs sur les images en niveaux de gris.
        #   Le modèle ne supporte que les images RGB.
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Redimensionnement sans distorsion via l'ajout de bandes grises (letterbox).
        #   On peut aussi redimensionner directement si la distorsion est acceptable.
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Ajout de la dimension batch (batch_size = 1).
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   Passage de l'image dans le réseau pour obtenir les prédictions brutes.
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   Empilement des boîtes prédites puis suppression des non-maximaux (NMS).
            #---------------------------------------------------------#
            outputs = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if outputs[0] is None: 
                return outputs

            top_label   = np.array(outputs[0][:, 5], dtype = 'int32')
            top_conf    = outputs[0][:, 4]
            top_boxes   = outputs[0][:, :4]

        for i, c in enumerate(top_label):
            result                      = {}
            top, left, bottom, right    = top_boxes[i]

            result["image_id"]      = int(image_id)
            result["category_id"]   = clsid2catid[c]
            result["bbox"]          = [float(left),float(top),float(right-left),float(bottom-top)]
            result["score"]         = float(top_conf[i])
            results.append(result)
        return results

if __name__ == "__main__":
    if not os.path.exists(temp_save_path):
        os.makedirs(temp_save_path)

    cocoGt      = COCO(cocoGt_path)
    ids         = list(cocoGt.imgToAnns.keys())
    clsid2catid = cocoGt.getCatIds()

    if map_mode == 0 or map_mode == 1:
        yolo = mAP_YOLO(confidence = 0.001, nms_iou = 0.65)

        with open(os.path.join(temp_save_path, 'eval_results.json'),"w") as f:
            results = []
            for image_id in tqdm(ids):
                image_path  = os.path.join(dataset_img_path, cocoGt.loadImgs(image_id)[0]['file_name'])
                image       = Image.open(image_path)
                results     = yolo.detect_image(image_id, image, results, clsid2catid)
            json.dump(results, f)

    if map_mode == 0 or map_mode == 2:
        cocoDt      = cocoGt.loadRes(os.path.join(temp_save_path, 'eval_results.json'))
        cocoEval    = COCOeval(cocoGt, cocoDt, 'bbox') 
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        print("Get map done.")
