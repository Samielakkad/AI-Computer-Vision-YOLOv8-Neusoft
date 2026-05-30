# Coeur de la boucle d'entraînement : tout ce qui se passe à l'intérieur d'un epoch
# vit ici. train.py l'appelle à chaque itération epoch par epoch.
#
# Ce que cette fonction fait concrètement :
#   (1) Forward + loss + backward sur tout le train set — avec support fp16 via AMP
#       pour diviser la mémoire GPU par deux sans sacrifier la précision.
#   (2) Validation complète à la fin de chaque epoch — on switche le modèle en eval
#       mode et on mesure la val_loss pour savoir si on overfitte ou non.
#   (3) Affichage tqdm en temps réel : loss courante et lr à chaque batch,
#       pour détecter immédiatement un divergence ou un plateau.
#   (4) Sauvegarde conditionnelle des poids : toutes les save_period epochs,
#       à la dernière epoch, et surtout dès qu'on bat le meilleur val_loss vu jusqu'ici.
#   (5) EMA (Exponential Moving Average) des poids : c'est le modèle EMA,
#       pas le modèle brut, qu'on utilise pour la validation et les checkpoints —
#       les poids moyennés généralisent mieux, surtout en fin d'entraînement.
import os
import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(
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
    Epoch,
    cuda,
    fp16,
    scaler,
    save_period,
    save_dir,
    local_rank=0,
):
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print("Start Training")
        pbar = tqdm(
            total=epoch_step,
            desc=f"Epoch {epoch + 1}/{Epoch}",
            postfix=dict,
            mininterval=0.3,
        )

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, bboxes = batch  # déballage direct du batch : images + targets
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)

        # zero_grad AVANT backward — indispensable, sinon les gradients s'accumulent
        # d'un batch à l'autre et on ne descend plus le bon gradient.
        optimizer.zero_grad()
        if not fp16:
            # Chemin standard fp32 : forward → loss → backward → clip → step
            outputs = model_train(images)
            loss_value = yolo_loss(outputs, bboxes)
            loss_value.backward()
            # Grad clipping à max_norm=10 : évite les explosions de gradient
            # fréquentes avec les têtes de détection multi-échelle de YOLOv8.
            # Sans ça, un batch difficile peut détruire les poids en un seul step.
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)

            optimizer.step()
        else:
            # Chemin fp16 avec AMP : autocast gère la précision automatiquement,
            # le GradScaler compense le fait que les gradients fp16 peuvent underflow.
            from torch.cuda.amp import autocast

            with autocast():
                outputs = model_train(images)
                loss_value = yolo_loss(outputs, bboxes)

            scaler.scale(loss_value).backward()
            # unscale_ avant clip_grad_norm pour que la norme soit comparable
            # à la norme fp32 — sinon on clipe des valeurs déjà scalées.
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm(model_train.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

        # EMA : après chaque optimizer step on met à jour la moyenne mobile des poids.
        # L'EMA lisse les oscillations du training et donne un modèle plus stable.
        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(
                **{"loss": loss / (iteration + 1), "lr": get_lr(optimizer)}
            )
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print("Finish Training")
        print("Start Validation")
        pbar = tqdm(
            total=epoch_step_val,
            desc=f"Epoch {epoch + 1}/{Epoch}",
            postfix=dict,
            mininterval=0.3,
        )

    # Pour la validation on utilise les poids EMA si disponibles — c'est eux
    # qui seront sauvegardés comme checkpoint, donc autant les évaluer directement.
    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, bboxes = batch  # même déballage que pour le train set
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)

            optimizer.zero_grad()
            outputs = model_train_eval(images)
            loss_value = yolo_loss(outputs, bboxes)
        val_loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(
                **{"val_loss": val_loss / (iteration + 1), "lr": get_lr(optimizer)}
            )
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print("Finish Validation")
        # On logue train_loss et val_loss dans loss_history pour tracer les courbes
        # d'apprentissage après coup et diagnostiquer overfitting / underfitting.
        loss_history.append_loss(
            epoch + 1, loss / epoch_step, val_loss / epoch_step_val
        )

        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print("Epoch:" + str(epoch + 1) + "/" + str(Epoch))

        print(
            "Total Loss: %.3f || Val Loss: %.3f "
            % (loss / epoch_step, val_loss / epoch_step_val)
        )

        # C'est le state_dict EMA qu'on sauvegarde, pas le modèle brut.
        # Les poids EMA sont ceux qui généraliseront le mieux en inférence.
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        # Checkpoint périodique : toutes les save_period epochs et à la toute dernière.
        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            torch.save(
                save_state_dict,
                os.path.join(
                    save_dir,
                    "ep%3d-loss%.3f-val_loss%.3f.pth"
                    % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val),
                ),
            )

        # Sauvegarde du meilleur modèle : on écrase best_epoch_weights.pth dès qu'on
        # atteint un nouveau minimum de val_loss — c'est le critère d'early stopping
        # qu'on utilise pour choisir les poids finaux de déploiement.
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(
            loss_history.val_loss
        ):
            print("Save best model to best_epoch_weights.pth")
            torch.save(
                save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth")
            )

        # last_epoch_weights.pth : checkpoint de reprise, toujours à jour,
        # utile pour reprendre l'entraînement si la session est interrompue.
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
