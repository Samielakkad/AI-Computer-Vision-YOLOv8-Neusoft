# -------------------------------------#
#       Entraînement sur le dataset
# -------------------------------------#
import datetime
import os
from functools import partial
import config

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (
    Loss,
    ModelEMA,
    get_lr_scheduler,
    set_optimizer_lr,
    weights_init,
)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (
    download_weights,
    get_classes,
    seed_everything,
    show_config,
    worker_init_fn,
)
from utils.utils_fit import fit_one_epoch

"""
Quelques points importants à garder en tête avant de lancer l'entraînement :

1.  Format du dataset : le code attend du VOC pur — images .jpg + labels .xml.
    Les images n'ont pas besoin d'avoir une taille fixe, le resize se fait
    automatiquement à l'entrée. Les images en niveaux de gris sont converties
    en RGB sans intervention manuelle. Si les images ne sont pas en .jpg,
    il faut les convertir en batch avant de démarrer.

2.  Interpréter la loss : ce qui compte ce n'est pas la valeur absolue mais la
    tendance — la val loss doit descendre régulièrement. Quand elle se stabilise,
    le modèle a convergé. Une loss qui semble "grande" n'est pas forcément mauvaise,
    ça dépend entièrement de comment elle est calculée. Si on veut l'afficher plus
    proprement on peut diviser par 10 000 dans la fonction de loss, mais ça ne
    change rien à l'optimisation. Les logs sont sauvegardés dans logs/loss_%Y_%m_%d_%H_%M_%S/.

3.  Checkpoints : les poids sont sauvegardés dans logs/ à chaque save_period epochs.
    Attention à bien distinguer epoch (une passe complète sur le dataset) et step
    (une descente de gradient). Si on entraîne seulement quelques steps, rien n'est
    sauvegardé — il faut atteindre la fin d'un epoch.
"""
if __name__ == "__main__":
    # ---------------------------------#
    #   Cuda : passer à False si pas
    #   de GPU disponible sur la machine
    # ---------------------------------#
    Cuda = True
    # ----------------------------------------------#
    #   Seed : fixe l'aléatoire pour que chaque run
    #   indépendant produise exactement les mêmes
    #   résultats — indispensable pour reproduire
    # ----------------------------------------------#
    seed = 11
    # ---------------------------------------------------------------------#
    #   distributed : entraînement multi-GPU sur une seule machine.
    #   Les commandes terminal ne fonctionnent qu'en Ubuntu.
    #   Sous Windows on est forcé en mode DP (DataParallel), DDP non supporté.
    #
    #   Mode DP :
    #       distributed = False
    #       lancer avec : CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   Mode DDP :
    #       distributed = True
    #       lancer avec : CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    distributed = False
    # ---------------------------------------------------------------------#
    #   sync_bn : BatchNorm synchronisé entre les GPUs, utile en mode DDP
    #   uniquement — inutile sur un seul GPU
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16 : mixed-precision (float16 + float32).
    #   Réduit la VRAM d'environ moitié, nécessite PyTorch >= 1.7.1.
    #   Sur ce setup je l'ai laissé à False pour la stabilité.
    # ---------------------------------------------------------------------#
    fp16 = False
    # ---------------------------------------------------------------------#
    #   classes_path : pointe vers le .txt des classes dans model_data/.
    #   À modifier absolument pour correspondre à son propre dataset
    #   avant tout lancement.
    # ---------------------------------------------------------------------#
    classes_path = config.vocClassesPath
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   Backbone YOLOv8-s pré-entraîné : on part TOUJOURS de là, jamais de zéro.
    #   Les features apprises sur ImageNet/COCO sont universelles — edges, textures,
    #   formes — elles se transfèrent très bien sur n'importe quel dataset de détection.
    #   Sans ces poids le backbone est aléatoire et le réseau n'apprend quasiment
    #   rien sur un dataset de cette taille.
    #
    #   Si l'entraînement a été interrompu, on peut reprendre en pointant model_path
    #   vers un checkpoint sauvegardé dans logs/ — les epochs reprennent là où on
    #   s'est arrêté, il faut juste ajuster Init_Epoch en conséquence.
    #
    #   model_path = '' désactive le chargement des poids (déconseillé sauf cas
    #   très particuliers avec énormément de données).
    #
    #   Pour entraîner depuis zéro (vraiment déconseillé) :
    #   model_path = '', Freeze_Train = False. Le réseau part de poids aléatoires
    #   et il faudra beaucoup plus d'epochs pour converger — et le résultat final
    #   sera souvent moins bon qu'avec pré-entraînement. Deux options si on insiste :
    #   1. Mosaic fort + UnFreeze_Epoch >= 300 + batch >= 16 + dataset > 10k images.
    #   2. Pré-entraîner d'abord un classifieur sur ImageNet, récupérer ses poids
    #      de backbone, puis les transférer ici.
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = config.pretrained_pth_Path
    # ------------------------------------------------------#
    #   input_shape : taille d'entrée du réseau.
    #   Doit être un multiple de 32 — contrainte des
    #   strides du backbone (8, 16, 32).
    # ------------------------------------------------------#
    input_shape = [640, 640]
    # ------------------------------------------------------#
    #   phi : variante de YOLOv8 à utiliser.
    #       n : yolov8_n  (nano, le plus léger)
    #       s : yolov8_s  (small, bon compromis)
    #       m : yolov8_m
    #       l : yolov8_l
    #       x : yolov8_x  (le plus lourd, le plus précis)
    #   Ici on utilise 's' — taille raisonnable pour le GPU
    #   disponible pendant le stage.
    # ------------------------------------------------------#
    phi = "s"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained : charge uniquement les poids du backbone (pas la tête).
    #   N'a de sens que si model_path est vide — sinon model_path prend la
    #   priorité et pretrained est ignoré.
    #   Si model_path = '' et pretrained = True  → backbone pré-entraîné, tête aléatoire.
    #   Si model_path = '' et pretrained = False → tout aléatoire (vraiment pas recommandé).
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = False
    # ------------------------------------------------------------------#
    #   mosaic : augmentation mosaïque — colle 4 images ensemble pour
    #   créer des scènes artificielles variées. Très efficace pour
    #   améliorer la détection de petits objets et diversifier les
    #   contextes visuels.
    #   mosaic_prob : probabilité d'appliquer mosaic à chaque step (50 %).
    #
    #   mixup : mélange linéaire de deux images mosaïquées.
    #   Ne s'active que si mosaic = True — ça n'a pas de sens sur une
    #   image non-mosaïquée dans ce contexte.
    #   mixup_prob : probabilité de mixup après mosaic (50 %).
    #   → probabilité totale mixup = mosaic_prob * mixup_prob = 25 %.
    #
    #   special_aug_ratio : comme dans YOLOX, les images mosaïquées
    #   s'éloignent beaucoup de la distribution naturelle. On limite donc
    #   mosaic aux special_aug_ratio premiers epochs (ici 70 %) pour laisser
    #   le modèle affiner sur des images réalistes en fin d'entraînement.
    # ------------------------------------------------------------------#
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7
    # ------------------------------------------------------------------#
    #   label_smoothing : adoucit les targets 0/1 vers ε/(K-1) et 1-ε.
    #   Évite la sur-confiance. En général ≤ 0.01 suffit.
    #   Laissé à 0 ici — le dataset est propre et le modèle n'a pas
    #   tendance à sur-fitter sur les labels dans ce setup.
    # ------------------------------------------------------------------#
    label_smoothing = 0

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   Stratégie deux phases — freeze puis unfreeze :
    #   Phase 1 (freeze) : on gèle le backbone, seule la tête est entraînée.
    #   La VRAM consommée est faible car pas de gradient dans le backbone.
    #   Utile quand le GPU est limité, et ça évite de "casser" les features
    #   pré-entraînées trop tôt.
    #
    #   Phase 2 (unfreeze) : tout le réseau est dégelé, le backbone peut
    #   s'adapter au dataset. Plus gourmand en VRAM mais indispensable pour
    #   que le backbone se spécialise vraiment sur les features du dataset cible.
    #
    #   Recommandations selon le cas :
    #   (a) Depuis les poids pré-entraînés complets :
    #       Adam  → Init_lr=1e-3, Freeze_Epoch=50, UnFreeze_Epoch=100, weight_decay=0
    #       SGD   → Init_lr=1e-2, Freeze_Epoch=50, UnFreeze_Epoch=300, weight_decay=5e-4
    #       UnFreeze_Epoch peut aller de 100 à 300.
    #   (b) Depuis zéro :
    #       Init_Epoch=0, UnFreeze_Epoch >= 300, Unfreeze_batch_size >= 16,
    #       Freeze_Train=False, optimizer SGD, mosaic=True.
    #   (c) Batch size :
    #       Le plus grand possible dans la VRAM disponible.
    #       BatchNorm impose batch_size >= 2.
    #       Freeze_batch_size ≈ 1–2× Unfreeze_batch_size — ne pas mettre un
    #       écart trop grand car ça affecte le scaling automatique du learning rate.
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Paramètres de la phase freeze (backbone gelé)
    #   Init_Epoch : epoch de départ. Si Init_Epoch > Freeze_Epoch,
    #               la phase freeze est sautée — pratique pour reprendre
    #               un entraînement interrompu à mi-chemin.
    #   Freeze_Epoch : fin de la phase freeze (ignoré si Freeze_Train=False)
    #   Freeze_batch_size : batch size pendant le freeze (on peut se permettre
    #                       plus grand car moins de gradients à stocker)
    # ------------------------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 32
    # ------------------------------------------------------------------#
    #   Paramètres de la phase unfreeze (backbone dégelé)
    #   UnFreeze_Epoch : nombre total d'epochs d'entraînement.
    #                    SGD converge plus lentement qu'Adam → besoin de
    #                    plus d'epochs. Adam peut s'en tirer avec moins.
    #   Unfreeze_batch_size : batch size une fois le backbone dégelé.
    #                         Généralement plus petit qu'en freeze car les
    #                         gradients remontent dans tout le réseau.
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = 30
    Unfreeze_batch_size = 16
    # ------------------------------------------------------------------#
    #   Freeze_Train : active la stratégie deux phases.
    #   True → freeze d'abord, unfreeze ensuite.
    #   False → on part directement en unfreeze sur tout le réseau.
    # ------------------------------------------------------------------#
    Freeze_Train = True

    # ------------------------------------------------------------------#
    #   Paramètres d'optimisation : learning rate, optimizer, scheduler
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr : learning rate maximum (au pic du warmup).
    #   Min_lr  : learning rate minimum en fin de cosine schedule.
    #             Par défaut 1 % du Init_lr — on ne descend pas à zéro
    #             pour garder un peu de dynamique même en fin d'entraînement.
    # ------------------------------------------------------------------#
    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type : 'sgd' ou 'adam'.
    #   SGD + momentum + cosine schedule → meilleurs résultats en général
    #   sur YOLO si on a assez d'epochs. Adam converge plus vite mais
    #   il ne faut pas utiliser weight_decay avec Adam (ça biaise
    #   l'estimation du second moment).
    #   momentum = 0.937 : valeur YOLOv5/v8 de référence pour SGD avec Nesterov.
    #   weight_decay = 5e-4 : régularisation L2 pour limiter l'overfitting.
    # ------------------------------------------------------------------#
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4
    # ------------------------------------------------------------------#
    #   lr_decay_type : type de scheduler.
    #   'cos' → cosine annealing — descente douce et progressive du
    #           learning rate, meilleure convergence que le step decay
    #           brutal. C'est le standard sur les détecteurs modernes.
    #   'step' → disponible mais moins utilisé ici.
    # ------------------------------------------------------------------#
    lr_decay_type = "cos"
    # ------------------------------------------------------------------#
    #   save_period : fréquence de sauvegarde des checkpoints en epochs.
    #   Toutes les 10 epochs on écrit un .pth dans logs/.
    # ------------------------------------------------------------------#
    save_period = 10
    # ------------------------------------------------------------------#
    #   save_dir : dossier de sauvegarde des poids et des logs de loss.
    # ------------------------------------------------------------------#
    save_dir = "logs"
    # ------------------------------------------------------------------#
    #   eval_flag   : active l'évaluation mAP sur le val set pendant
    #                 l'entraînement.
    #   eval_period : tous les combien d'epochs on évalue.
    #                 L'évaluation prend du temps — trop fréquente elle
    #                 ralentit vraiment l'entraînement. Toutes les 10
    #                 epochs c'est un bon compromis.
    #   Note : le mAP obtenu ici sera légèrement différent de get_map.py
    #   car (1) c'est calculé sur le val set et (2) les paramètres
    #   d'évaluation sont volontairement conservateurs pour aller vite.
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period = 10
    # ------------------------------------------------------------------#
    #   num_workers : nombre de processus parallèles pour le chargement
    #   des données. Plus c'est grand, plus le DataLoader est rapide,
    #   mais ça consomme de la RAM. Sur une machine avec peu de RAM,
    #   mettre 2 ou même 0.
    # ------------------------------------------------------------------#
    # num_workers         = 4
    num_workers = 4
    # ------------------------------------------------------#
    #   train_annotation_path : chemin vers le .txt listant
    #                           les images et labels d'entraînement
    #   val_annotation_path   : idem pour la validation
    # ------------------------------------------------------#
    train_annotation_path = config.train_annotation_path
    val_annotation_path = config.val_annotation_path

    seed_everything(seed)
    # ------------------------------------------------------#
    #   Détection et configuration des GPUs disponibles
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(
                f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training..."
            )
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        rank = 0

    # ------------------------------------------------------#
    #   Chargement des noms de classes et du nombre de classes
    # ------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    # ----------------------------------------------------#
    #   Téléchargement des poids pré-entraînés si nécessaire
    # ----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)
            dist.barrier()
        else:
            download_weights(phi)

    # ------------------------------------------------------#
    #   Instanciation du modèle YOLOv8
    # ------------------------------------------------------#
    model = YoloBody(input_shape, num_classes, phi, pretrained=pretrained)

    if model_path != "":
        # ------------------------------------------------------#
        #   Chargement du fichier de poids
        # ------------------------------------------------------#
        if local_rank == 0:
            print("Load weights {}.".format(model_path))

        # ------------------------------------------------------#
        #   Chargement sélectif par correspondance de clés et
        #   de shapes — les couches qui ne matchent pas sont
        #   ignorées sans planter (utile quand la tête change).
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   Affichage des clés qui n'ont pas pu être chargées —
        #   normal que la head ne charge pas (nombre de classes
        #   différent), mais le backbone doit charger intégralement.
        # ------------------------------------------------------#
        if local_rank == 0:
            print(
                "\nSuccessful Load Key:",
                str(load_key)[:500],
                "……\nSuccessful Load Key Num:",
                len(load_key),
            )
            print(
                "\nFail To Load Key:",
                str(no_load_key)[:500],
                "……\nFail To Load Key num:",
                len(no_load_key),
            )
            print(
                "\n\033[1;33;44mNote : il est normal que la tête (head) ne soit pas chargée. En revanche, si le Backbone n'est pas chargé, c'est une erreur.\033[0m"
            )

    # ----------------------#
    #   Initialisation de la fonction de loss
    # ----------------------#
    yolo_loss = Loss(model)
    # ----------------------#
    #   Initialisation du logger de loss (tensorboard + csv)
    # ----------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(
            datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S"
        )
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ------------------------------------------------------------------#
    #   fp16 / AMP : PyTorch < 1.2 ne supporte pas GradScaler.
    #   Si on est en dessous de 1.7.1 on peut voir "could not be resolved"
    #   — dans ce cas forcer fp16 = False.
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    # ----------------------------#
    #   SyncBatchNorm : synchronise les stats BN entre GPUs en DDP.
    #   Inutile sur un seul GPU ou en mode DP.
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   Mode DDP : chaque process prend un GPU, communication NCCL
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train, device_ids=[local_rank], find_unused_parameters=True
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ----------------------------#
    #   EMA (Exponential Moving Average) : maintient une copie lissée
    #   des poids du modèle. Les poids EMA varient moins d'un batch
    #   à l'autre — ça stabilise les prédictions en validation et
    #   donne généralement un meilleur mAP que les poids "bruts".
    # ----------------------------#
    ema = ModelEMA(model_train)

    # ---------------------------#
    #   Lecture des fichiers d'annotations (un chemin d'image par ligne)
    # ---------------------------#
    with open(train_annotation_path, encoding="utf-8") as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding="utf-8") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path=classes_path,
            model_path=model_path,
            input_shape=input_shape,
            Init_Epoch=Init_Epoch,
            Freeze_Epoch=Freeze_Epoch,
            UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size,
            Unfreeze_batch_size=Unfreeze_batch_size,
            Freeze_Train=Freeze_Train,
            Init_lr=Init_lr,
            Min_lr=Min_lr,
            optimizer_type=optimizer_type,
            momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period,
            save_dir=save_dir,
            num_workers=num_workers,
            num_train=num_train,
            num_val=num_val,
        )
        # ---------------------------------------------------------#
        #   Vérification que le nombre total de steps est suffisant.
        #   SGD a besoin d'environ 50 000 steps pour bien converger,
        #   Adam peut s'en tirer avec 15 000. Si le dataset est petit
        #   ou UnFreeze_Epoch trop bas, on affiche un warning avec
        #   la valeur d'epoch recommandée. Seule la phase unfreeze
        #   est prise en compte dans ce calcul.
        # ----------------------------------------------------------#
        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError("Dataset trop petit pour entraîner — il faut l'agrandir.")
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print(
                "\n\033[1;33;44m[Warning] Avec l'optimiseur %s, il est conseillé de viser un nombre total de steps supérieur à %d.\033[0m"
                % (optimizer_type, wanted_step)
            )
            print(
                "\033[1;33;44m[Warning] Pour ce run : %d images d'entraînement, Unfreeze_batch_size = %d, %d epochs au total, soit un nombre total de steps calculé de %d.\033[0m"
                % (num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step)
            )
            print(
                "\033[1;33;44m[Warning] Le nombre total de steps (%d) est inférieur au minimum conseillé (%d) — il vaudrait mieux fixer le nombre total d'epochs à %d.\033[0m"
                % (total_step, wanted_step, wanted_epoch)
            )

    # ------------------------------------------------------#
    #   Le backbone pré-entraîné extrait des features génériques
    #   qui n'ont pas besoin d'être retouchées tout de suite.
    #   On gèle d'abord le backbone pour entraîner uniquement
    #   la tête de détection — ça va vite et ça évite de corrompre
    #   les features dès le début. Ensuite on dégèle pour que
    #   le backbone s'adapte au dataset cible.
    #   Si OOM (out of memory) → réduire Batch_size.
    # ------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        # ------------------------------------#
        #   Gel du backbone pour la phase freeze

        # ------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # -------------------------------------------------------------------#
        #   Si pas de freeze, on part directement avec Unfreeze_batch_size
        # -------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # -------------------------------------------------------------------#
        #   Scaling adaptatif du learning rate selon le batch size réel.
        #   Formule : lr_fit = batch_size / nbs * Init_lr
        #   nbs = 64 est le "batch size de référence" pour lequel Init_lr
        #   a été calibré. Si on tourne avec batch=16 au lieu de 64, le
        #   gradient est 4× plus bruité — on scale le lr en proportion.
        #   On clamp ensuite entre lr_limit_min et lr_limit_max pour éviter
        #   les valeurs absurdes sur des très petits ou très grands batchs.
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == "adam" else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(
            max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2
        )

        # ---------------------------------------#
        #   Construction de l'optimizer.
        #   On sépare les paramètres en 3 groupes :
        #   pg0 : poids BatchNorm (pas de weight_decay sur BN)
        #   pg1 : autres poids (weight_decay appliqué)
        #   pg2 : biais (pas de weight_decay sur les biais)
        #   C'est la même stratégie que YOLOv5 officiel.
        # ---------------------------------------#
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            "adam": optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            "sgd": optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True),
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        # ---------------------------------------#
        #   Cosine annealing scheduler : le learning rate descend
        #   en suivant une demi-cosinus de Init_lr_fit à Min_lr_fit
        #   sur la totalité des UnFreeze_Epoch epochs.
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(
            lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch
        )

        # ---------------------------------------#
        #   Nombre de steps par epoch (entier)
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("Dataset trop petit pour continuer l'entraînement — il faut l'agrandir.")

        if ema:
            ema.updates = epoch_step * Init_Epoch

        # ---------------------------------------#
        #   DataLoaders train et val.
        #   Val : mosaic et mixup désactivés — on veut évaluer
        #   sur des images réelles, pas des compositions artificielles.
        # ---------------------------------------#
        train_dataset = YoloDataset(
            train_lines,
            input_shape,
            num_classes,
            epoch_length=UnFreeze_Epoch,
            mosaic=mosaic,
            mixup=mixup,
            mosaic_prob=mosaic_prob,
            mixup_prob=mixup_prob,
            train=True,
            special_aug_ratio=special_aug_ratio,
        )
        val_dataset = YoloDataset(
            val_lines,
            input_shape,
            num_classes,
            epoch_length=UnFreeze_Epoch,
            mosaic=False,
            mixup=False,
            mosaic_prob=0,
            mixup_prob=0,
            train=False,
            special_aug_ratio=0,
        )

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                shuffle=True,
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                shuffle=False,
            )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(
            train_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=yolo_dataset_collate,
            sampler=train_sampler,
            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
        )
        gen_val = DataLoader(
            val_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=yolo_dataset_collate,
            sampler=val_sampler,
            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
        )

        # ----------------------#
        #   Callback d'évaluation : calcule et enregistre la courbe mAP
        # ----------------------#
        if local_rank == 0:
            eval_callback = EvalCallback(
                model,
                input_shape,
                class_names,
                num_classes,
                val_lines,
                log_dir,
                Cuda,
                eval_flag=eval_flag,
                period=eval_period,
            )
        else:
            eval_callback = None

        # ---------------------------------------#
        #   Boucle d'entraînement principale
        # ---------------------------------------#
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # ---------------------------------------#
            #   Passage en phase unfreeze : si on a atteint Freeze_Epoch
            #   et que le backbone est encore gelé, on le dégèle.
            #   On recalcule le lr adaptatif avec le nouveau batch_size,
            #   on recrée le scheduler et les DataLoaders.
            # ---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                # -------------------------------------------------------------------#
                #   Re-scaling du learning rate pour le nouveau batch_size (unfreeze)
                # -------------------------------------------------------------------#
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == "adam" else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == "adam" else 5e-4
                Init_lr_fit = min(
                    max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max
                )
                Min_lr_fit = min(
                    max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2),
                    lr_limit_max * 1e-2,
                )
                # ---------------------------------------#
                #   Nouveau scheduler cosine pour la phase unfreeze
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(
                    lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch
                )

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset trop petit pour continuer l'entraînement — il faut l'agrandir.")

                if ema:
                    ema.updates = epoch_step * epoch

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(
                    train_dataset,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=yolo_dataset_collate,
                    sampler=train_sampler,
                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                )
                gen_val = DataLoader(
                    val_dataset,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=yolo_dataset_collate,
                    sampler=val_sampler,
                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
                )

                UnFreeze_flag = True

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(
                model_train,
                model,
                ema,
                yolo_loss,
                loss_history,
                eval_callback,
                optimizer,
                epoch,
                epoch_step,
                epoch_step_val,
                gen,
                gen_val,
                UnFreeze_Epoch,
                Cuda,
                fp16,
                scaler,
                save_period,
                save_dir,
                local_rank,
            )

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
