#-----------------------------------------------------------------------#
#   predict.py regroupe en un seul fichier tous les modes d'utilisation :
#   prédiction image par image, détection vidéo/webcam, benchmark FPS,
#   parcours de répertoire, heatmap et export ONNX.
#   Il suffit de changer la variable `mode` pour switcher.
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   `mode` contrôle le comportement du script :
    #   'predict'           Prédiction sur une image unique. Pour sauvegarder le résultat,
    #                       ajoute r_image.save("img.jpg") dans le bloc correspondant.
    #   'video'             Détection sur flux vidéo ou webcam (video_path=0 pour la webcam).
    #   'fps'               Benchmark FPS sur fps_image_path — moyenne sur test_interval inférences.
    #   'dir_predict'       Parcourt dir_origin_path et sauvegarde les résultats dans dir_save_path.
    #   'heatmap'           Visualise les cartes d'activation du réseau (où il "regarde") sous forme
    #                       de heatmap colorée superposée à l'image d'entrée.
    #   'export_onnx'       Exporte le modèle au format ONNX (requiert PyTorch >= 1.7.1).
    #----------------------------------------------------------------------------------------------------------#
    mode = "heatmap"
    #-------------------------------------------------------------------------#
    #   crop    : si True, découpe et sauvegarde chaque objet détecté
    #             dans img_crop/ après la prédiction.
    #   count   : si True, affiche le nombre de détections par classe.
    #   crop et count ne sont actifs qu'en mode 'predict'.
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          Chemin vers la vidéo, ou 0 pour la webcam.
    #                       Exemple : video_path = "xxx.mp4" lit le fichier à la racine du projet.
    #   video_save_path     Chemin de sauvegarde de la vidéo traitée. "" = pas de sauvegarde.
    #                       Exemple : video_save_path = "yyy.mp4".
    #   video_fps           FPS de la vidéo sauvegardée.
    #
    #   Ces trois variables ne sont actives qu'en mode 'video'.
    #   Pour terminer proprement et finaliser l'écriture, faire Ctrl+C ou laisser la vidéo se terminer.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       Nombre de forward passes pour le benchmark FPS :
    #                       plus c'est grand, plus la mesure est stable (GPU warmup inclus avant).
    #   fps_image_path      Image utilisée pour le benchmark.
    #
    #   Actifs uniquement en mode 'fps'.
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     Dossier source contenant les images à traiter.
    #   dir_save_path       Dossier de destination pour les images annotées.
    #
    #   Actifs uniquement en mode 'dir_predict'.
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   Chemin de sauvegarde de la heatmap générée.
    #
    #   Actif uniquement en mode 'heatmap'.
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    #   simplify        Applique onnx-simplifier pour réduire le graphe ONNX
    #                   (fold constants, éliminer noeuds redondants).
    #   onnx_save_path  Chemin de sauvegarde du fichier ONNX exporté.
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":
        '''
        Quelques customisations utiles en mode predict :
        1. Pour sauvegarder l'image annotée : r_image.save("img.jpg") après detect_image.
        2. Pour récupérer les coordonnées des boxes : lire top, left, bottom, right dans
           yolo.detect_image, dans la section de dessin.
        3. Pour cropper les objets détectés : utiliser les mêmes coordonnées et image.crop().
        4. Pour compter un type d'objet précis (ex: voitures) : dans detect_image, tester
           if predicted_class == 'car' et incrémenter un compteur, puis afficher avec draw.text.
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Impossible de lire la caméra ou la vidéo — vérifie que la webcam est connectée ou que le chemin vidéo est correct.")

        fps = 0.0
        while(True):
            t1 = time.time()
            # Lecture de la frame courante
            ref, frame = capture.read()
            if not ref:
                break
            # Conversion BGR → RGB (OpenCV lit en BGR, PIL/YOLO attendent du RGB)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Conversion numpy array → PIL Image pour detect_image
            frame = Image.fromarray(np.uint8(frame))
            # Inférence YOLO sur la frame
            frame = np.array(yolo.detect_image(frame))
            # Reconversion RGB → BGR pour l'affichage OpenCV
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)

    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
