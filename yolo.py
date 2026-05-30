import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox

'''
Commentaires importants si tu entraînes sur ton propre dataset !
'''
class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   Si tu utilises tes propres poids entraînés, modifie OBLIGATOIREMENT
        #   model_path et classes_path — sinon le réseau charge les mauvaises classes.
        #   model_path pointe vers le fichier .pth dans logs/, classes_path vers le
        #   .txt dans model_data/.
        #
        #   Après l'entraînement, plusieurs checkpoints s'accumulent dans logs/ :
        #   prends celui dont la perte sur le val set est la plus basse.
        #   Attention : perte val basse ≠ mAP max — c'est juste un bon indicateur
        #   de généralisation. Si les shapes ne matchent pas au chargement, vérifie
        #   que model_path et classes_path correspondent bien au run d'entraînement.
        #--------------------------------------------------------------------------#
        "model_path"        : 'model_data/best_epoch_weights.pth',
        "classes_path"      : 'model_data/coco_classes.txt',
        #---------------------------------------------------------------------#
        #   Taille d'entrée du réseau — doit être un multiple de 32 à cause
        #   des downsampling successifs dans le backbone (stride 8/16/32).
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #------------------------------------------------------#
        #   Variante de YOLOv8 à utiliser :
        #   n : yolov8_n  (nano, le plus léger)
        #   s : yolov8_s
        #   m : yolov8_m
        #   l : yolov8_l
        #   x : yolov8_x  (le plus puissant)
        #------------------------------------------------------#
        "phi"               : 's',
        #---------------------------------------------------------------------#
        #   Seuil de confidence : seules les bounding boxes dont le score
        #   dépasse cette valeur sont conservées avant le NMS.
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   IoU threshold pour le NMS — plus c'est bas, plus on supprime
        #   de boxes qui se chevauchent. 0.3 est un bon compromis sur COCO.
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   letterbox_image : resize sans déformation en ajoutant des bandes
        #   grises pour conserver le ratio d'aspect original.
        #   Désactiver ça et faire un resize direct donne parfois de meilleurs
        #   résultats en pratique (objets moins distordus dans certains cas).
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   Passe à False si tu n'as
        #   pas de GPU disponible.
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initialisation de la classe YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

        #---------------------------------------------------#
        #   Charge les noms de classes et déduit num_classes,
        #   puis instancie DecodeBox qui gère le DFL decode
        #   et le NMS en aval.
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.bbox_util                      = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))

        #---------------------------------------------------#
        #   Génère une couleur HSV distincte par classe,
        #   convertie en RGB entier pour PIL.
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    #---------------------------------------------------#
    #   Construit l'architecture et charge les poids.
    #   fuse() fusionne Conv+BN avant l'inférence pour
    #   réduire la latence — on ne fuse jamais pendant
    #   l'entraînement car BN a besoin de ses stats séparées.
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #---------------------------------------------------#
        #   Instancie YoloBody et charge les poids entraînés.
        #---------------------------------------------------#
        self.net    = YoloBody(self.input_shape, self.num_classes, self.phi)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   Pipeline complet d'inférence sur une image PIL :
    #   prétraitement → forward → DFL decode → NMS → dessin
    #---------------------------------------------------#
    def detect_image(self, image, crop = False, count = False):
        #---------------------------------------------------#
        #   Récupère (H, W) de l'image originale pour
        #   recalibrer les bounding boxes après NMS.
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Force la conversion en RGB : PIL peut ouvrir des PNG
        #   RGBA ou des images en niveaux de gris, ce qui ferait
        #   planter les couches Conv qui attendent 3 canaux.
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Letterbox resize : on remplit les bords avec du gris
        #   pour atteindre input_shape sans écraser le ratio.
        #   Sans ça, un objet large et fin serait écrasé et le
        #   réseau le reconnaîtrait moins bien.
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   h, w, 3 → transpose → 3, h, w → expand → 1, 3, h, w
        #   Normalisation dans preprocess_input (÷255, centrage).
        #   Le batch_size=1 est nécessaire pour le forward pass.
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   Forward pass : le réseau retourne les prédictions brutes
            #   sous forme de tenseurs DFL (Distribution Focal Loss).
            #---------------------------------------------------------#
            outputs = self.net(images)
            #---------------------------------------------------------#
            #   decode_box décode les sorties DFL en coordonnées (x,y,w,h)
            #   puis les ramène dans l'espace de l'image originale.
            #---------------------------------------------------------#
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   NMS : supprime les bounding boxes redondantes en ne
            #   gardant que celle avec le score max dans chaque cluster
            #   de boxes qui se chevauchent (IoU > nms_iou).
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)

            if results[0] is None:
                return image

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   Taille de police proportionnelle à la hauteur de l'image,
        #   épaisseur du rectangle proportionnelle à la résolution.
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        #---------------------------------------------------------#
        #   Mode comptage : affiche combien de fois chaque classe
        #   apparaît dans les détections après NMS.
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   Mode crop : découpe et sauvegarde chaque objet détecté
        #   dans img_crop/ pour analyse ou dataset curation.
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   Dessin des bounding boxes et labels sur l'image PIL.
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            bbox = draw.textbbox((0,0),label,font=font)
            label_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Même conversion RGB que dans detect_image : obligatoire
        #   pour éviter une erreur de shape sur les images grises.
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Letterbox resize pour conserver le ratio d'aspect
        #   et rester cohérent avec les conditions d'entraînement.
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Ajout de la dimension batch : h, w, 3 → 1, 3, h, w
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   Passe de chauffe : le premier forward est toujours plus
            #   lent (allocations CUDA, compilation JIT). On ne chronomètre
            #   pas ce run.
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   NMS inclus dans la mesure de FPS car il fait partie
            #   du pipeline réel de détection.
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   Forward pass mesuré : on répète test_interval fois
                #   pour moyenner et lisser les variations GPU.
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   NMS compris dans la boucle chronométrée.
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                            image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        #---------------------------------------------------------#
        #   Conversion RGB comme partout ailleurs — la heatmap
        #   utilise aussi le même pipeline de prétraitement.
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Letterbox resize : cohérence avec l'inférence normale.
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   1, 3, h, w — même shape que pour detect_image.
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   On récupère les feature maps intermédiaires x (avant
            #   decode DFL) pour visualiser où le réseau "regarde".
            #   On isole la partie classe (les num_classes derniers
            #   canaux) en ignorant les canaux de régression de boîte.
            #---------------------------------------------------------#
            dbox, cls, x, anchors, strides = self.net(images)
            outputs = [xi.split((xi.size()[1] - self.num_classes, self.num_classes), 1)[1] for xi in x]

        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask    = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            #---------------------------------------------------------#
            #   Reshape + sigmoid pour obtenir des probabilités de classe
            #   par position spatiale, puis on garde le max sur les classes.
            #   Résultat : une carte 2D H×W de "confiance" par pixel.
            #   On redimensionne à la taille originale et on accumule
            #   le maximum pour fusionner les 3 échelles de détection.
            #---------------------------------------------------------#
            sub_output = np.transpose(np.reshape(sub_output, [b, -1, h, w]), [0, 2, 3, 1])[0]
            score      = np.max(sigmoid(sub_output[..., :]), -1)
            score      = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score    = (score * 255).astype('uint8')
            mask            = np.maximum(mask, normed_score)

        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]

        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8')
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Même conversion RGB — une image en niveaux de gris
        #   ou RGBA ferait planter le réseau en inférence.
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Letterbox resize pour rester cohérent avec les
        #   conditions d'entraînement lors du calcul du mAP.
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   1, 3, h, w — ajout de la dimension batch.
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   Forward + decode DFL → coordonnées de bounding boxes.
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   NMS pour ne garder qu'une box par objet — nécessaire
            #   pour que le calcul mAP ne double-compte pas les détections.
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape,
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)

            if results[0] is None:
                return

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return
