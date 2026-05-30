import random

import numpy as np
import torch
from PIL import Image


#---------------------------------------------------------#
#   On force la conversion en RGB avant toute inférence.
#   Le réseau attend exactement 3 canaux — passer une image
#   en niveaux de gris ou RGBA ferait crasher le forward pass.
#   Toutes les images passent ici, peu importe leur format
#   d'origine.
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

#---------------------------------------------------#
#   Resize de l'image d'entrée vers la taille cible.
#   Si letterbox_image est True, on préserve le ratio
#   d'aspect en ajoutant des bandes grises (128, 128, 128)
#   sur les bords — c'est ce qui évite de déformer les
#   objets et de fausser les coordonnées des bboxes.
#   Si False, on fait un resize direct, plus rapide mais
#   qui écrase le ratio : acceptable sur des datasets
#   très homogènes, risqué sinon.
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   Lecture du fichier de classes : une ligne = une
#   classe. On renvoie la liste et le nombre total,
#   les deux étant utilisés partout dans le réseau.
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   Lecture du learning rate courant depuis le
#   premier param_group de l'optimizer. Utile pour
#   logger le lr réel après scheduler.
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   Fixe tous les seeds (Python, NumPy, PyTorch CPU
#   et GPU) pour garantir la reproductibilité des
#   expériences. Sans ça, deux runs sur le même
#   dataset donnent des résultats différents et on
#   ne peut pas comparer proprement les ablations.
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   Seed par worker du DataLoader : chaque worker
#   reçoit un seed dérivé du rank global + seed de
#   base, ce qui évite que tous les workers génèrent
#   les mêmes augmentations aléatoires en parallèle.
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(phi, model_dir="./model_data"):
    import os

    from torch.hub import load_state_dict_from_url

    download_urls = {
        "n" : './yolov8_n_backbone_weights.pth',
        "s" : './yolov8_s_backbone_weights.pth',
        "m" : './yolov8_m_backbone_weights.pth',
        "l" : './yolov8_l_backbone_weights.pth',
        "x" : './yolov8_x_backbone_weights.pth',
    }
    url = download_urls[phi]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
