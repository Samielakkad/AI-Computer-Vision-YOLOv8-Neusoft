"""
Calcul du mAP adapté du projet open-source Cartucho/mAP et des outils pycocotools.
Utilisé ici pour évaluer le mAP VOC@0.5 et le mAP COCO@0.5:0.95 du détecteur YOLOv8.
Aucune logique originale n'a été modifiée — seuls les commentaires ont été traduits.
"""
import glob
import json
import math
import operator
import os
import shutil
import sys

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except:
    pass
import cv2
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        Taux de manqués moyen en échelle logarithmique (log-average miss rate) :
        calculé en moyennant le taux de manqués sur 9 points FPPI équidistants
        dans l'espace logarithmique entre 10e-2 et 10e0.

        Sorties :
        lamr  | taux de manqués moyen logarithmique
        mr    | taux de manqués
        fppi  | faux positifs par image
    """

    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


"""
 Lève une erreur et quitte le programme.
"""


def error(msg):
    print(msg)
    sys.exit(0)


"""
 Vérifie si la valeur est un flottant strictement compris entre 0.0 et 1.0.
"""


# Tentative de conversion : on essaie d'abord de convertir value en flottant via float(value).
# Vérification de la plage : on s'assure que val est strictement supérieur à 0.0 et inférieur à 1.0.
# Gestion des exceptions : si la conversion échoue (ValueError), la fonction retourne False.
def is_float_between_0_and_1(value):
    try:
        val = float(value)
        # 0.0 et 1.0 représentent respectivement les flottants 0 et 1. On vérifie que la valeur est dans l'intervalle ouvert (0, 1).
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False


"""
 Calcule l'AP (Average Precision) à partir des tableaux de rappel et de précision.
    1) On calcule une version de la courbe précision/rappel rendue monotone décroissante.
    2) L'AP est ensuite obtenu par intégration numérique de l'aire sous cette courbe.
"""


# Calcule la Précision Moyenne (AP) pour une tâche de détection ou de classification.
# Paramètres :
# rec  : liste (ou tableau) de valeurs de rappel — proportion de vrais positifs parmi tous les positifs réels.
# prec : liste (ou tableau) de valeurs de précision — proportion de vrais positifs parmi toutes les détections positives.
def voc_ap(rec, prec):
    """
    --- Code MATLAB officiel VOC2012 ---
        mrec=[0 ; rec ; 1] ;
        mpre=[0 ; prec ; 0] ;
        for i = numel(mpre)-1 : -1 : 1
            mpre(i) = max(mpre(i), mpre(i+1)) ;
        end
        i = find(mrec(2:end) ~= mrec(1:end-1)) + 1 ;
        ap = sum((mrec(i) - mrec(i-1)) .* mpre(i)) ;
    """
    # Insère 0.0 au début de rec pour délimiter la courbe
    rec.insert(0, 0.0)  # ajoute la borne inférieure du rappel
    # Ajoute 1.0 à la fin de rec pour fermer la courbe
    rec.append(1.0)  # ajoute la borne supérieure du rappel
    # Copie superficielle de rec dans mrec
    mrec = rec[:]  # copie de travail du vecteur rappel
    # Insère 0.0 au début de prec (précision initiale nulle)
    prec.insert(0, 0.0)  # borne inférieure de la précision
    # Ajoute 0.0 à la fin de prec (précision finale nulle)
    prec.append(0.0)  # borne supérieure de la précision
    mpre = prec[:]  # copie de travail de la précision, rendue ensuite monotone décroissante
    """
     On rend la précision monotone décroissante
        (de la fin vers le début)
        matlab : for i = numel(mpre)-1 : -1 : 1
                     mpre(i) = max(mpre(i), mpre(i+1)) ;
    """
    for i in range(len(mpre) - 2, -1, -1):  # on parcourt de l'avant-dernier élément vers le début
        mpre[i] = max(mpre[i], mpre[i + 1])  # garantit que la précision ne remonte jamais
    """
     On construit la liste des indices où le rappel change de valeur.
        matlab : i = find(mrec(2:end) ~= mrec(1:end-1)) + 1 ;
    """
    i_list = []
    for i in range(1, len(mrec)):  # on parcourt mrec à partir du deuxième élément
        if mrec[i] != mrec[i - 1]:  # si le rappel courant diffère du précédent
            i_list.append(i)  # if it was matlab would be i + 1 # on enregistre l'indice (Python base-0, MATLAB base-1)
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    # ap   -- valeur de la précision moyenne
    # mrec -- liste de rappel modifiée (avec bornes)
    # mpre -- liste de précision modifiée (avec bornes, rendue monotone décroissante)
    ap = 0.0
    for i in i_list:
        # mrec[i]-mrec[i-1] est la largeur de l'intervalle de rappel courant.
        # mpre[i] est la précision maximale sur cet intervalle (courbe monotone décroissante).
        # Le produit (mrec[i]-mrec[i-1])*mpre[i] donne l'aire du rectangle correspondant
        # sous la courbe précision/rappel.
        # La somme de ces aires sur tous les intervalles donne l'AP approximé.
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""


def file_lines_to_list(path):
    # open txt file lines to a list
    # Ouverture du fichier avec `with` pour garantir sa fermeture automatique.
    # path est le chemin vers le fichier texte.
    with open(path) as f:
        # Lecture de toutes les lignes ; readlines conserve les caractères de saut de ligne.
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    # On supprime les espaces/sauts de ligne en début et fin de chaque ligne via strip().
    content = [x.strip() for x in content]
    return content


"""
 Draws text in image
"""


def draw_text_in_image(img, text, pos, color, line_width):
    # Police HERSHEY_PLAIN : police simple fournie par OpenCV.
    font = cv2.FONT_HERSHEY_PLAIN
    # Échelle de la police : 1 correspond à la taille originale.
    fontScale = 1
    # Type de trait : 1 équivaut à LINE_8 (pas d'anticrénelage ici).
    lineType = 1
    # Position de départ du texte (coin inférieur gauche).
    bottomLeftCornerOfText = pos
    # Tracé du texte sur l'image.
    # Arguments : image, texte, position, police, échelle, couleur, type de trait.
    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
    # Récupération de la largeur du texte rendu (la hauteur est ignorée ici).
    text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
    # Retourne l'image modifiée et la largeur cumulée (utile pour aligner des blocs de texte).
    return img, (line_width + text_width)


"""
 Plot - adjust axes
"""


# adjust_axes élargit la plage de l'axe X pour que les étiquettes textuelles ne soient pas tronquées.
# r     : objet renderer (matplotlib) — utilisé pour obtenir la bounding box du texte.
# t     : objet matplotlib.text.Text représentant l'étiquette à afficher.
# fig   : objet matplotlib.figure.Figure — fournit la largeur courante et le DPI.
# axes  : objet matplotlib.axes.Axes — permet de lire et modifier la plage de l'axe X.
def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    # Bounding box du texte dans le renderer — sert à mesurer sa largeur en pixels.
    bb = t.get_window_extent(renderer=r)
    # Conversion de la largeur du texte de pixels en pouces (division par le DPI).
    text_width_inches = bb.width / fig.dpi
    # Largeur courante de la figure en pouces.
    current_fig_width = fig.get_figwidth()
    # Nouvelle largeur nécessaire pour contenir le texte.
    new_fig_width = current_fig_width + text_width_inches
    # Rapport entre nouvelle et ancienne largeur — utilisé pour redimensionner l'axe X.
    propotion = new_fig_width / current_fig_width
    # get axis limit
    # Plage courante de l'axe X.
    x_lim = axes.get_xlim()
    # On étend la borne droite de l'axe X proportionnellement pour éviter que le texte
    # déborde à droite du graphique.
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


"""
 Trace un graphique avec Matplotlib.
"""


# Trace un diagramme en barres horizontales à partir d'un dictionnaire de valeurs.
# Paramètres : dictionnaire de valeurs, nombre de classes, titre de la fenêtre, titre du graphique,
#              étiquette de l'axe X, chemin de sortie, affichage à l'écran, couleur, dict de vrais positifs.
def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                   true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    # Tri du dictionnaire par valeur croissante via operator.itemgetter — résultat : liste de tuples.
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    # Décompactage de la liste de tuples en deux listes séparées : clés et valeurs.
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    # Si un dictionnaire de vrais positifs est fourni, on effectue un tracé bicolore TP/FP.
    if true_p_bar != "":
        """
         Cas bicolore :
            - Vert  -> TP : vrai positif (objet détecté et correspondant à une vérité terrain)
            - Rouge -> FP : faux positif (objet détecté mais ne correspondant à aucune vérité terrain)
            - Orange-> FN : faux négatif (objet présent dans la vérité terrain mais non détecté)
        """
        # Initialisation des listes FP et TP.
        fp_sorted = []
        tp_sorted = []
        # On calcule FP et TP pour chaque classe triée.
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        # Tracé des barres FP (rouge).
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        # Tracé des barres TP (vert), empilées à la suite des FP via le paramètre `left`.
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                 left=fp_sorted)
        # add legend
        # Ajout de la légende.
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        # Affichage des valeurs numériques à côté des barres.
        fig = plt.gcf()  # gcf - get current figure # récupère la figure courante
        axes = plt.gca()  # récupère l'axe courant
        r = fig.canvas.get_renderer()  # récupère le renderer pour mesurer le texte
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            # Astuce bicolore : on trace d'abord la chaîne complète en vert, puis on repeint la partie FP en rouge.
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            # On n'ajuste l'axe X que pour la plus grande barre (la dernière de la liste triée).
            # L'indice de ce dernier élément est len(sorted_values)-1 car Python est base-0.
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        # Cas simple : pas de dict TP fourni — on trace un seul diagramme en barres uniforme.
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Affiche les valeurs numériques à côté de chaque barre.
        """
        # Récupération de la figure, des axes et du renderer.
        fig = plt.gcf()  # gcf - get current figure # figure courante
        axes = plt.gca()  # axes courants
        r = fig.canvas.get_renderer()  # renderer pour mesurer le texte
        # Parcours des valeurs triées pour annoter chaque barre.
        for i, val in enumerate(sorted_values):
            # Conversion en chaîne avec un espace initial pour l'espacement visuel.
            str_val = " " + str(val)  # add a space before # espace de confort visuel
            # Pour les valeurs < 1 (ex. AP en proportion), on formate à 2 décimales.
            if val < 1.0:
                str_val = " {0:.2f}".format(val)  # formatage à 2 décimales pour les valeurs fractionnaires
            # Tracé de l'étiquette textuelle à côté de la barre.
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # Pour la plus grande barre (dernière de la liste), on ajuste l'axe X si nécessaire.
            # re-set axes to show number inside the figure
            if i == (len(sorted_values) - 1):  # largest bar
                adjust_axes(r, t, fig, axes)  # ajustement de l'axe pour éviter les débordements
    # set window title
    # Définition du titre de la fenêtre.
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    # Affichage des noms de classes comme étiquettes de l'axe Y.
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)  # étiquettes de l'axe Y avec taille de police
    """
     Redimensionnement de la hauteur de la figure en fonction du contenu.
    """
    # Hauteur initiale de la figure (en pouces).
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    # DPI de la figure — sert à convertir des points en pouces.
    dpi = fig.dpi
    # Hauteur totale requise en points : nombre de classes × taille de police × facteur d'espacement.
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    # Conversion de la hauteur requise de points en pouces.
    height_in = height_pt / dpi
    # compute the required figure height
    # Marges en pourcentage de la hauteur totale de la figure.
    top_margin = 0.15  # in percentage of the figure height # marge haute (15 %)
    bottom_margin = 0.05  # in percentage of the figure height # marge basse (5 %)
    # La hauteur du contenu utile représente (1 - marges) de la hauteur totale.
    # On divise donc height_in par ce facteur pour obtenir la hauteur totale nécessaire,
    # marges comprises.
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    # On agrandit la figure uniquement si la hauteur calculée dépasse la hauteur actuelle.
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    # Définition du titre du graphique.
    plt.title(plot_title, fontsize=14)  # titre avec taille de police 14
    # set axis titles
    # plt.xlabel('classes')
    # Définition du titre de l'axe X.
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    # Ajustement automatique de la disposition des sous-graphiques.
    fig.tight_layout()
    # save the plot
    # Sauvegarde du graphique dans le fichier de sortie.
    fig.savefig(output_path)
    # show image
    # Affichage interactif si demandé.
    if to_show:
        plt.show()
    # close the plot
    # Fermeture de la figure pour libérer la mémoire.
    plt.close()


# Calcule et visualise les métriques de performance du détecteur : AP, F1, rappel, précision.
# Compare les résultats de détection (DR) aux annotations de vérité terrain (GT).
# Si IMG_PATH existe, génère une animation montrant chaque correspondance détection/GT.
def get_map(MINOVERLAP, draw_plot, score_threhold=0.5, path='./map_out'):
    GT_PATH = os.path.join(path, 'ground-truth')  # chemin vers les annotations de vérité terrain
    DR_PATH = os.path.join(path, 'detection-results')
    IMG_PATH = os.path.join(path, 'images-optional')
    TEMP_FILES_PATH = os.path.join(path, '.temp_files')
    RESULTS_FILES_PATH = os.path.join(path, 'results')
    # Détermine si le mode animation est activé (images optionnelles présentes).
    show_animation = True
    # Préparation des répertoires temporaires et de résultats.
    if os.path.exists(IMG_PATH):
        for dirpath, dirnames, files in os.walk(IMG_PATH):
            if not files:
                show_animation = False
    else:
        show_animation = False
    # Création du répertoire temporaire si inexistant.
    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)

    if os.path.exists(RESULTS_FILES_PATH):
        shutil.rmtree(RESULTS_FILES_PATH)
    else:
        os.makedirs(RESULTS_FILES_PATH)
    # Création des sous-répertoires pour les graphiques si le tracé est activé.
    if draw_plot:
        try:
            matplotlib.use('TkAgg')
        except:
            pass
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "AP"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "F1"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Recall"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Precision"))
    if show_animation:
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "images", "detections_one_by_one"))
    # Lecture et traitement des fichiers d'annotations de vérité terrain.
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # Initialisation des compteurs de classes et d'images.
    gt_counter_per_class = {}
    counter_images_per_class = {}
    # Parcours des fichiers GT — parse les boîtes englobantes et les sauvegarde en JSON temporaire.
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:  # parcours ligne par ligne du fichier GT
            try:
                # Format attendu : "class_name left top right bottom [difficult]"
                if "difficult" in line:  # la ligne contient le marqueur "difficult"
                    class_name, left, top, right, bottom, _difficult = line.split()  # décomposition directe
                    is_difficult = True  # marquage comme échantillon difficile
                else:
                    class_name, left, top, right, bottom = line.split()  # décomposition sans marqueur "difficult"
            # Si la décomposition directe échoue (nom de classe multi-mots, etc.)
            except:
                if "difficult" in line:  # la ligne contient toujours "difficult"
                    line_split = line.split()  # découpage en tokens
                    # On lit les coordonnées depuis la fin de la liste (ordre inverse).
                    _difficult = line_split[-1]  # dernier token : marqueur "difficult"
                    bottom = line_split[-2]  # avant-dernier : coordonnée bottom
                    right = line_split[-3]  # coordonnée right
                    top = line_split[-4]  # coordonnée top
                    left = line_split[-5]  # coordonnée left
                    # Le nom de classe peut contenir des espaces — on reconstitue depuis les tokens restants.
                    class_name = ""
                    for name in line_split[:-5]:  # tokens avant les 5 derniers
                        class_name += name + " "  # concaténation avec espace
                    class_name = class_name[:-1]  # suppression de l'espace final
                    is_difficult = True  # marquage comme échantillon difficile
                else:
                    # Même logique sans le marqueur "difficult".
                    line_split = line.split()
                    bottom = line_split[-1]  # coordonnée bottom
                    right = line_split[-2]  # coordonnée right
                    top = line_split[-3]  # coordonnée top
                    left = line_split[-4]  # coordonnée left
                    class_name = ""
                    for name in line_split[:-4]:  # tokens avant les 4 derniers
                        class_name += name + " "  # concaténation
                    class_name = class_name[:-1]  # suppression de l'espace final
                # Construction de la chaîne de boîte englobante : "left top right bottom"
            bbox = left + " " + top + " " + right + " " + bottom
            # Traitement selon que l'échantillon est difficile ou non.
            if is_difficult:
                # Boîte difficile : enregistrée mais non comptée dans le mAP.
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                # Réinitialisation du marqueur pour la prochaine itération.
                is_difficult = False
            else:
                # Boîte normale : enregistrée et comptée.
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                # Incrémentation du compteur de la classe concernée.
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # Première occurrence de cette classe : initialisation à 1.
                    gt_counter_per_class[class_name] = 1
                # Si la classe n'a pas encore été vue pour cette image, on met à jour le compteur d'images.
                if class_name not in already_seen_classes:
                    # La classe existe déjà dans le compteur d'images : on incrémente.
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # Première image pour cette classe : initialisation à 1.
                        counter_images_per_class[class_name] = 1
                    # On marque la classe comme déjà vue pour cette image.
                    already_seen_classes.append(class_name)
            # Sauvegarde de la liste de boîtes englobantes dans un fichier JSON temporaire.
        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)
    # Extraction et tri de toutes les classes présentes dans les annotations GT.
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)  # tri alphabétique des classes
    # Nombre total de classes.
    n_classes = len(gt_classes)

    # Recherche et tri des fichiers de résultats de détection.
    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()
    # Parcours de chaque classe pour regrouper les détections.
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []  # liste de boîtes pour la classe courante
        for txt_file in dr_files_list:
            # Extraction de l'identifiant du fichier (sans extension).
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            # Chemin vers le fichier GT correspondant (vérifié uniquement pour la première classe).
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            # Pour la première classe, on vérifie que le fichier GT existe bien.
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error(error_msg)  # appel de la fonction error définie plus haut
            # Lecture des lignes du fichier de détection courant.
            lines = file_lines_to_list(txt_file)
            # Parcours ligne par ligne.
            for line in lines:
                try:
                    # Format attendu : "class_name confidence left top right bottom"
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except:
                    # Décomposition de secours en cas de nom de classe multi-mots.
                    line_split = line.split()  # découpage en tokens
                    # Lecture des coordonnées et de la confiance depuis la fin de la liste.
                    bottom = line_split[-1]  # coordonnée bottom
                    right = line_split[-2]  # coordonnée right
                    top = line_split[-3]  # coordonnée top
                    left = line_split[-4]  # coordonnée left
                    confidence = line_split[-5]  # score de confiance
                    tmp_class_name = ""
                    for name in line_split[:-5]:
                        tmp_class_name += name + " "
                    tmp_class_name = tmp_class_name[:-1]
                # On ne conserve que les détections correspondant à la classe courante.
                if tmp_class_name == class_name:
                    # Construction et enregistrement de la boîte englobante.
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
            # Tri des détections par confiance décroissante.
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        # Sauvegarde des détections triées dans un fichier JSON temporaire.
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)
    # Initialisation de la somme des AP à 0.
    sum_AP = 0.0
    # Dictionnaire pour stocker l'AP de chaque classe.
    ap_dictionary = {}
    # Dictionnaire pour stocker le LAMR (log-average miss rate) de chaque classe.
    lamr_dictionary = {}
    # Ouverture du fichier de résultats en écriture.
    with open(RESULTS_FILES_PATH + "/results.txt", 'w') as results_file:
        # En-tête du fichier de résultats.
        results_file.write("# AP and precision/recall per class\n")
        # Dictionnaire pour compter les vrais positifs (TP) par classe.
        count_true_positives = {}

        # Parcours de toutes les classes présentes dans la vérité terrain.
        for class_index, class_name in enumerate(gt_classes):
            # Initialisation du compteur TP à 0 pour la classe courante.
            count_true_positives[class_name] = 0
            # Chemin vers le fichier JSON de détections pour cette classe.
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            # Chargement des données de détection.
            dr_data = json.load(open(dr_file))

            # Nombre total de détections pour cette classe.
            nd = len(dr_data)
            # Initialisation des tableaux TP, FP et scores.
            tp = [0] * nd
            fp = [0] * nd
            score = [0] * nd
            # Indice de la dernière détection dont le score dépasse le seuil.
            score_threhold_idx = 0
            # Parcours de toutes les détections.
            for idx, detection in enumerate(dr_data):
                # Identifiant du fichier et score de confiance de la détection courante.
                file_id = detection["file_id"]
                score[idx] = float(detection["confidence"])
                # Mise à jour de l'indice de seuil si la confiance est suffisante.
                if score[idx] >= score_threhold:
                    score_threhold_idx = idx  # mise à jour de l'indice seuil

                # Mode animation : chargement de l'image correspondante.
                if show_animation:
                    # Recherche de l'image de vérité terrain correspondant à ce file_id.
                    ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                    # Aucune image trouvée : on lève une erreur.
                    if len(ground_truth_img) == 0:
                        error("Error. Image not found with id: " + file_id)
                    # Plusieurs images trouvées pour le même id — cas improbable en pratique.
                    elif len(ground_truth_img) > 1:
                        error("Error. Multiple image with id: " + file_id)
                    else:
                        # Lecture de l'image via OpenCV.
                        img = cv2.imread(IMG_PATH + "/" + ground_truth_img[
                            0])  # on accède au premier (et unique) fichier image associé à la détection courante
                        # Chemin vers l'image cumulative (annotations superposées au fil des itérations).
                        img_cumulative_path = RESULTS_FILES_PATH + "/images/" + ground_truth_img[0]
                        # Chargement de l'image cumulative existante ou copie de l'image originale.
                        if os.path.isfile(img_cumulative_path):
                            img_cumulative = cv2.imread(img_cumulative_path)
                        else:
                            img_cumulative = img.copy()
                        # Ajout d'une bordure noire en bas pour afficher les informations textuelles.
                        bottom_border = 60
                        # Couleur de la bordure en format BGR (convention OpenCV).
                        BLACK = [0, 0, 0]
                        # Application de la bordure inférieure noire à l'image.
                        img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                    # Chemin vers le fichier JSON de vérité terrain pour ce fichier.
                gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                # Chargement des données de vérité terrain.
                ground_truth_data = json.load(open(gt_file))
                # Initialisation du meilleur IoU à -1 (aucun chevauchement trouvé).
                ovmax = -1
                # Initialisation de l'objet GT correspondant à -1 (aucune correspondance).
                gt_match = -1
                # Conversion des coordonnées de la boîte détectée en flottants.
                bb = [float(x) for x in detection["bbox"].split()]
                # Parcours de tous les objets de la vérité terrain.
                for obj in ground_truth_data:
                    # On ne compare qu'avec les objets de la même classe.
                    if obj["class_name"] == class_name:
                        # Conversion des coordonnées de la boîte GT en flottants.
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        # Calcul de la zone d'intersection entre la boîte détectée et la boîte GT.
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        # Calcul de la largeur et de la hauteur de l'intersection.
                        # Le +1 convertit des coordonnées entières (intervalle fermé) en dimensions pixel.
                        # bi[0], bi[2] : abscisses gauche/droite ; bi[1], bi[3] : ordonnées haut/bas.
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        # L'intersection n'est réelle que si largeur et hauteur sont positives.
                        if iw > 0 and ih > 0:
                            # Aire de l'union = somme des deux aires moins l'intersection.
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                              + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            # Calcul de l'IoU (Intersection over Union).
                            ov = iw * ih / ua
                            # Mise à jour du meilleur IoU et de la boîte GT correspondante.
                            if ov > ovmax:
                                ovmax = ov  # nouveau meilleur IoU
                                gt_match = obj  # objet GT associé
                if show_animation:
                    status = "NO MATCH FOUND!"  # état initial : aucune correspondance
                # Seuil IoU minimal pour valider une détection comme TP.
                min_overlap = MINOVERLAP
                # Si le meilleur IoU dépasse le seuil, on vérifie si c'est un vrai positif.
                if ovmax >= min_overlap:
                    # La boîte GT ne doit pas être difficile et ne doit pas avoir déjà été utilisée.
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            # La détection est un vrai positif (TP).
                            tp[idx] = 1  # 1 = TP, 0 = FP
                            # On marque la boîte GT comme utilisée pour éviter les doubles comptages.
                            gt_match["used"] = True
                            # Incrémentation du compteur TP pour cette classe.
                            count_true_positives[class_name] += 1
                            # Mise à jour du fichier GT pour refléter l'utilisation de la boîte.
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                            # Mise à jour du statut d'animation si activé.
                            if show_animation:
                                status = "MATCH!"
                        else:
                            # La boîte GT a déjà été utilisée : la détection est un faux positif (FP).
                            fp[idx] = 1
                            # En animation, on signale une correspondance répétée.
                            if show_animation:
                                status = "REPEATED MATCH!"
                else:
                    # L'IoU est inférieur au seuil minimal : la détection est un faux positif (FP).
                    fp[idx] = 1
                    # Si l'IoU est positif mais sous le seuil, on signale un chevauchement insuffisant.
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

                """
                Tracé de l'image pour l'animation frame par frame.
                """
                # Mode animation : superposition des informations sur l'image.
                if show_animation:
                    # Dimensions de l'image.
                    height, widht = img.shape[:2]
                    # Couleurs pour les annotations textuelles.
                    white = (255, 255, 255)
                    light_blue = (255, 200, 100)
                    green = (0, 255, 0)
                    light_red = (30, 30, 255)
                    margin = 10  # marge en pixels par rapport au bord de l'image
                    # 1nd line
                    v_pos = int(height - margin - (bottom_border / 2.0))  # position verticale de la 1re ligne
                    text = "Image: " + ground_truth_img[0] + " "
                    # Tracé de la 1re ligne et récupération de la largeur occupée.
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                    # Tracé du nom de classe à la suite, décalé de la largeur précédente.
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue,
                                                         line_width)
                    if ovmax != -1:  # une intersection a été trouvée
                        color = light_red
                        # Texte et couleur selon que l'IoU est suffisant ou non.
                        if status == "INSUFFICIENT OVERLAP":
                            text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                        else:
                            text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                            color = green
                        # Tracé de l'information IoU sur l'image.
                        img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                    # 2nd line
                    # Position verticale de la 2e ligne.
                    v_pos += int(bottom_border / 2.0)
                    rank_pos = str(idx + 1)  # rang de la détection (base 1)
                    text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(
                        float(detection["confidence"]) * 100)
                    # Tracé de la 2e ligne (rang et confiance).
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    color = light_red
                    if status == "MATCH!":
                        color = green  # correspondance trouvée : on passe en vert
                    text = "Result: " + status + " "
                    # Tracé du statut de correspondance.
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                    # Tracé des boîtes englobantes GT et détectées.
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if ovmax > 0:
                        bbgt = [int(round(float(x))) for x in gt_match["bbox"].split()]
                        # Tracé de la boîte GT sur l'image courante et l'image cumulative.
                        cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                        cv2.rectangle(img_cumulative, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 2)
                        cv2.putText(img_cumulative, class_name, (bbgt[0], bbgt[1] - 5), font, 0.6, light_blue, 1,
                                    cv2.LINE_AA)
                    bb = [int(i) for i in bb]
                    # Tracé de la boîte détectée sur l'image courante et l'image cumulative.
                    cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                    cv2.rectangle(img_cumulative, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                    cv2.putText(img_cumulative, class_name, (bb[0], bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                    # Affichage de l'image dans la fenêtre d'animation.
                    cv2.imshow("Animation", img)
                    cv2.waitKey(20)  # pause de 20 ms entre chaque frame
                    # Sauvegarde de l'image annotée.
                    output_img_path = RESULTS_FILES_PATH + "/images/detections_one_by_one/" + class_name + "_detection" + str(
                        idx) + ".jpg"
                    cv2.imwrite(output_img_path, img)
                    cv2.imwrite(img_cumulative_path, img_cumulative)

            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val

            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val

            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name], 1)

            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]), 1)

            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            F1 = np.array(rec) * np.array(prec) * 2 / np.where((np.array(prec) + np.array(rec)) == 0, 1,
                                                               (np.array(prec) + np.array(rec)))

            sum_AP += ap
            text = "{0:.2f}%".format(
                ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)

            if len(prec) > 0:
                F1_text = "{0:.2f}".format(F1[score_threhold_idx]) + " = " + class_name + " F1 "
                Recall_text = "{0:.2f}%".format(rec[score_threhold_idx] * 100) + " = " + class_name + " Recall "
                Precision_text = "{0:.2f}%".format(prec[score_threhold_idx] * 100) + " = " + class_name + " Precision "
            else:
                F1_text = "0.00" + " = " + class_name + " F1 "
                Recall_text = "0.00%" + " = " + class_name + " Recall "
                Precision_text = "0.00%" + " = " + class_name + " Precision "

            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")

            if len(prec) > 0:
                print(text + "\t||\tscore_threhold=" + str(score_threhold) + " : " + "F1=" + "{0:.2f}".format(
                    F1[score_threhold_idx]) \
                      + " ; Recall=" + "{0:.2f}%".format(
                    rec[score_threhold_idx] * 100) + " ; Precision=" + "{0:.2f}%".format(
                    prec[score_threhold_idx] * 100))
            else:
                print(text + "\t||\tscore_threhold=" + str(
                    score_threhold) + " : " + "F1=0.00% ; Recall=0.00% ; Precision=0.00%")
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)
            lamr_dictionary[class_name] = lamr

            if draw_plot:
                plt.plot(rec, prec, '-o')
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

                fig = plt.gcf()
                fig.canvas.set_window_title('AP ' + class_name)

                plt.title('class: ' + text)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(RESULTS_FILES_PATH + "/AP/" + class_name + ".png")
                plt.cla()

                plt.plot(score, F1, "-", color='orangered')
                plt.title('class: ' + F1_text + "\nscore_threhold=" + str(score_threhold))
                plt.xlabel('Score_Threhold')
                plt.ylabel('F1')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(RESULTS_FILES_PATH + "/F1/" + class_name + ".png")
                plt.cla()

                plt.plot(score, rec, "-H", color='gold')
                plt.title('class: ' + Recall_text + "\nscore_threhold=" + str(score_threhold))
                plt.xlabel('Score_Threhold')
                plt.ylabel('Recall')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(RESULTS_FILES_PATH + "/Recall/" + class_name + ".png")
                plt.cla()

                plt.plot(score, prec, "-s", color='palevioletred')
                plt.title('class: ' + Precision_text + "\nscore_threhold=" + str(score_threhold))
                plt.xlabel('Score_Threhold')
                plt.ylabel('Precision')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(RESULTS_FILES_PATH + "/Precision/" + class_name + ".png")
                plt.cla()
        # Fermeture de toutes les fenêtres OpenCV si le mode animation était actif.
        if show_animation:
            cv2.destroyAllWindows()
        # Si aucune classe n'a été détectée, on affiche un message d'erreur et on retourne 0.
        if n_classes == 0:
            print("Aucune classe détectée. Vérifiez les labels et le paramètre classes_path dans get_map.py.")
            return 0
        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP * 100)
        results_file.write(text + "\n")
        print(text)

    shutil.rmtree(
        TEMP_FILES_PATH)  # suppression récursive du répertoire temporaire et de son contenu

    """
    Comptage total des détections par classe.
    """
    det_counter_per_class = {}
    for txt_file in dr_files_list:
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                det_counter_per_class[class_name] = 1
    dr_classes = list(det_counter_per_class.keys())

    """
    Écriture dans results.txt du nombre d'objets de vérité terrain par classe.
    """
    with open(RESULTS_FILES_PATH + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    """
    Finalisation du comptage des vrais positifs — initialisation à 0 pour les classes non présentes en GT.
    """
    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    """
    Écriture dans results.txt du nombre de détections par classe (avec TP et FP).
    """
    with open(RESULTS_FILES_PATH + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            results_file.write(text)

    """
    Tracé du nombre total d'occurrences de chaque classe dans la vérité terrain.
    """
    if draw_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = RESULTS_FILES_PATH + "/ground-truth-info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
        )

    # """
    # Plot the total number of occurences of each class in the "detection-results" folder
    # """
    # if draw_plot:
    #     window_title = "detection-results-info"
    #     # Plot title
    #     plot_title = "detection-results\n"
    #     plot_title += "(" + str(len(dr_files_list)) + " files and "
    #     count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
    #     plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    #     # end Plot title
    #     x_label = "Number of objects per class"
    #     output_path = RESULTS_FILES_PATH + "/detection-results-info.png"
    #     to_show = False
    #     plot_color = 'forestgreen'
    #     true_p_bar = count_true_positives
    #     draw_plot_func(
    #         det_counter_per_class,
    #         len(det_counter_per_class),
    #         window_title,
    #         plot_title,
    #         x_label,
    #         output_path,
    #         to_show,
    #         plot_color,
    #         true_p_bar
    #         )

    """
    Tracé du graphique LAMR (log-average miss rate) par classe, en ordre décroissant.
    """
    if draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = RESULTS_FILES_PATH + "/lamr.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
        )

    """
    Tracé du graphique mAP (AP par classe, en ordre décroissant).
    """
    if draw_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP * 100)
        x_label = "Average Precision"
        output_path = RESULTS_FILES_PATH + "/mAP.png"
        to_show = True
        plot_color = 'royalblue'
        draw_plot_func(
            ap_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
        )
    return mAP


def preprocess_gt(gt_path, class_names):
    image_ids = os.listdir(gt_path)
    results = {}

    images = []
    bboxes = []
    for i, image_id in enumerate(image_ids):
        lines_list = file_lines_to_list(os.path.join(gt_path, image_id))
        boxes_per_image = []
        image = {}
        image_id = os.path.splitext(image_id)[0]
        image['file_name'] = image_id + '.jpg'
        image['width'] = 1
        image['height'] = 1
        # -----------------------------------------------------------------#
        #
        #   Résout le problème 'Results do not correspond to current coco set'
        #   en utilisant l'identifiant de fichier (chaîne) comme clé d'image.
        # -----------------------------------------------------------------#
        image['id'] = str(image_id)

        for line in lines_list:
            difficult = 0
            if "difficult" in line:
                line_split = line.split()
                left, top, right, bottom, _difficult = line_split[-5:]
                class_name = ""
                for name in line_split[:-5]:
                    class_name += name + " "
                class_name = class_name[:-1]
                difficult = 1
            else:
                line_split = line.split()
                left, top, right, bottom = line_split[-4:]
                class_name = ""
                for name in line_split[:-4]:
                    class_name += name + " "
                class_name = class_name[:-1]

            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            if class_name not in class_names:
                continue
            cls_id = class_names.index(class_name) + 1
            bbox = [left, top, right - left, bottom - top, difficult, str(image_id), cls_id,
                    (right - left) * (bottom - top) - 10.0]
            boxes_per_image.append(bbox)
        images.append(image)
        bboxes.extend(boxes_per_image)
    results['images'] = images

    categories = []
    for i, cls in enumerate(class_names):
        category = {}
        category['supercategory'] = cls
        category['name'] = cls
        category['id'] = i + 1
        categories.append(category)
    results['categories'] = categories

    annotations = []
    for i, box in enumerate(bboxes):
        annotation = {}
        annotation['area'] = box[-1]
        annotation['category_id'] = box[-2]
        annotation['image_id'] = box[-3]
        annotation['iscrowd'] = box[-4]
        annotation['bbox'] = box[:4]
        annotation['id'] = i
        annotations.append(annotation)
    results['annotations'] = annotations
    return results


def preprocess_dr(dr_path, class_names):
    image_ids = os.listdir(dr_path)
    results = []
    for image_id in image_ids:
        lines_list = file_lines_to_list(os.path.join(dr_path, image_id))
        image_id = os.path.splitext(image_id)[0]
        for line in lines_list:
            line_split = line.split()
            confidence, left, top, right, bottom = line_split[-5:]
            class_name = ""
            for name in line_split[:-5]:
                class_name += name + " "
            class_name = class_name[:-1]
            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            result = {}
            result["image_id"] = str(image_id)
            if class_name not in class_names:
                continue
            result["category_id"] = class_names.index(class_name) + 1
            result["bbox"] = [left, top, right - left, bottom - top]
            result["score"] = float(confidence)
            results.append(result)
    return results


def get_coco_map(class_names, path):
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    COCO_PATH = os.path.join(path, 'coco_eval')

    if not os.path.exists(COCO_PATH):
        os.makedirs(COCO_PATH)

    GT_JSON_PATH = os.path.join(COCO_PATH, 'instances_gt.json')
    DR_JSON_PATH = os.path.join(COCO_PATH, 'instances_dr.json')

    with open(GT_JSON_PATH, "w") as f:
        results_gt = preprocess_gt(GT_PATH, class_names)
        json.dump(results_gt, f, indent=4)

    with open(DR_JSON_PATH, "w") as f:
        results_dr = preprocess_dr(DR_PATH, class_names)
        json.dump(results_dr, f, indent=4)
        if len(results_dr) == 0:
            print("Aucun objet détecté.")
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    cocoGt = COCO(GT_JSON_PATH)
    cocoDt = cocoGt.loadRes(DR_JSON_PATH)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats