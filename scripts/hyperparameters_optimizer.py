# author: Louca Malerba
# date: 07/08/2025

# Public libraries importations
import sys
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import optuna
from functools import partial

# Private libraries importations-
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

import dataset
import train



#################################







if __name__ == "__main__":
    print("Starting hyperparameters optimization with Optuna...")

    # Configuration fixe (non optimisée par Optuna)
    NUM_CLASSES = 10
    IMAGE_WIDTH = 40
    IMAGE_LENGTH = 40
    CHANNELS = 3
    path_to_dataset = '/Users/loucamalerba/Desktop/captcha_dataset_detection/Vision_transformer_digits/digits_dataset/images'

    # Définition des transformations (hors de la fonction objective pour éviter de les recréer)
    # train_transform = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     transforms.Resize((IMAGE_LENGTH, IMAGE_WIDTH)),
    #     transforms.RandomRotation(10),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5), (0.5)),
    # ])

    # train_transform = transforms.Compose([
    #     transforms.Resize((IMAGE_LENGTH, IMAGE_WIDTH)),
    #     transforms.RandomRotation(10),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5), (0.5)),
    # ])

    # test_transform = transforms.Compose([
    #     transforms.Resize((IMAGE_LENGTH, IMAGE_WIDTH)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5), (0.5)),
    # ])

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_LENGTH, IMAGE_WIDTH)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(((0.5), (0.5), (0.5)), ((0.5), (0.5), (0.5))),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_LENGTH, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(((0.5), (0.5), (0.5)), ((0.5), (0.5), (0.5))),
    ])

    def objective(trial):
        # 1. Suggest hyperparameters with Optuna (avec contrainte embed_dim % num_heads == 0)
        BATCH_SIZE = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        EPOCHS = 75
        LEARNING_RATE = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        PATCH_WIDTH = PATCH_LENGTH = trial.suggest_int('patch_size', 4, 8, step=2)

        # D'abord num_heads, puis embed_dim compatible
        # Liste des num_heads possibles pour embed_dim dans [128, 256, 512]
        possible_num_heads = {
            128: [1, 2, 4, 8, 16, 32, 64],
            256: [1, 2, 4, 8, 16, 32, 64, 128],
            512: [1, 2, 4, 8, 16, 32, 64, 128, 256]
        }

        # Choisir d'abord embed_dim, puis num_heads compatible
        EMBED_DIM = trial.suggest_categorical('embed_dim', [128, 256, 512])
        NUM_HEADS = trial.suggest_categorical('num_heads', possible_num_heads[EMBED_DIM])

        DEPTH = trial.suggest_int('depth', 4, 12)
        MLP_DIM = trial.suggest_categorical('mlp_dim', [256, 512, 1024])
        DROP_RATE = trial.suggest_float('drop_rate', 0.0, 0.3)

        # 2. Create datasets (doit être dans la fonction pour éviter les fuites de mémoire)
        train_dataset = dataset.ImageCaptchaDataset(
            root=path_to_dataset,
            train=True,
            img_h=IMAGE_LENGTH,
            img_w=IMAGE_WIDTH,
            transform=train_transform
        )

        test_dataset = dataset.ImageCaptchaDataset(
            root=path_to_dataset,
            train=False,
            img_h=IMAGE_LENGTH,
            img_w=IMAGE_WIDTH,
            transform=test_transform
        )

        # 3. Create dataloaders
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        # 4. Configuration fixe pour l'optimiseur et la loss
        criterion = "CrossEntropyLoss"
        optimizer = "Adam"

        # 5. Appel à la fonction d'entraînement
        test_acc = train.train(
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            num_classes=NUM_CLASSES,
            image_width=IMAGE_WIDTH,
            image_length=IMAGE_LENGTH,
            patch_width=PATCH_WIDTH,
            patch_length=PATCH_LENGTH,
            channels=CHANNELS,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            depth=DEPTH,
            mlp_dim=MLP_DIM,
            drop_rate=DROP_RATE,
            train_transform=train_transform,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            save_folder='Vision_transformer_digits/checkpoints',
            save_plots=False  # Désactivé pour accélérer l'optimisation
        )

        return test_acc  # Optuna va maximiser cette valeur

    # Configuration de l'étude Optuna
    study = optuna.create_study(
        direction="maximize",  # On veut maximiser l'acc
        sampler=optuna.samplers.TPESampler(),  # Algorithme d'optimisation
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)  # Pour arrêter les mauvais essais tôt
    )

    # Lancement de l'optimisation
    study.optimize(
        objective,
        n_trials=50,  # Nombre d'essais
        timeout=3600  # Temps maximum en secondes (1 heure)
    )

    # Affichage des meilleurs résultats
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Test loss: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 6. Entraînement final avec les meilleurs hyperparamètres
    print("\nTraining final model with best hyperparameters...")
    best_params = study.best_params

    # Recréer les dataloaders avec le meilleur batch_size
    best_batch_size = best_params['batch_size']
    train_loader = DataLoader(dataset=train_dataset, batch_size=best_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=best_batch_size, shuffle=False)

    # Lancement de l'entraînement final (avec sauvegarde des plots cette fois)
    train.train(
        batch_size=best_batch_size,
        epochs=EPOCHS,
        learning_rate=best_params['learning_rate'],
        num_classes=NUM_CLASSES,
        image_width=IMAGE_WIDTH,
        image_length=IMAGE_LENGTH,
        patch_width=best_params['patch_size'],
        patch_length=best_params['patch_size'],
        channels=CHANNELS,
        embed_dim=best_params['embed_dim'],
        num_heads=best_params['num_heads'],
        depth=best_params['depth'],
        mlp_dim=best_params['mlp_dim'],
        drop_rate=best_params['drop_rate'],
        train_transform=train_transform,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        save_folder='Vision_transformer_digits/checkpoints',
        save_plots=True
    )
