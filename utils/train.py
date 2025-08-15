# author: Louca Malerba
# date: 07/08/2025

# Public libraries importations
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import sys

# Private libraries importations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

import dataset
from model import VisionTransformer  


#################################

def save_checkpoint(state, is_best, checkpoint_dir='./checkpoints'):
    """Sauvegarde un checkpoint (toujours last, si best marque aussi best)."""
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, 'model_last.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'model_best.pth')
        torch.save(state, best_filename)

def evaluate(model, loader, device='cpu'):
    model.eval() # Set the mode of the model into evlauation
    correct = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(dim=1) == y).sum().item()
    return correct / len(loader.dataset)

def train(batch_size,
        epochs,
        learning_rate,
        num_classes,
        image_width,
        image_length,
        patch_width,
        patch_length,
        channels,
        embed_dim,
        num_heads,
        depth,
        mlp_dim,
        drop_rate,
        train_transform,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        save_folder,
        device='cpu',
        save_plots=False):
    
    
    
    # 1. Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    # 2. Initialize model
    model = VisionTransformer(
            [image_width, image_length],
            [patch_width, patch_length],
            channels,
            num_classes,
            embed_dim,
            depth,
            num_heads,
            mlp_dim,
            drop_rate
        ).to(device)
    
    # 3. Extract from parameters
    if criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()

    if optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Model training
    print("Starting training...")
    _best_acc = 0
    train_accuracies, test_accuracies = [], []
    checkpoint_dir=f'{save_folder}/{batch_size}_{epochs}_{learning_rate}_{num_classes}_{image_width}_{image_length}_{patch_width}_{patch_length}_{channels}_{embed_dim}_{num_heads}_{depth}_{mlp_dim}_{drop_rate}'
    for epoch in tqdm(range(epochs)):
        ## Training phase
        # Set the mode of the model into training
        model.train()

        total_loss, correct = 0, 0
        for x, y in train_loader:
            # Moving (Sending) our data into the target device
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            # 1. Forward pass (model outputs raw logits)
            out = model(x)

            # 2. Calcualte loss (per batch)
            loss = criterion(out, y)
            # 3. Perform backpropgation
            loss.backward()
            # 4. Perforam Gradient Descent
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
        # You have to scale the loss (Normlization step to make the loss general across all batches)
        train_loss, train_acc =  total_loss / len(train_loader.dataset), correct / len(train_loader.dataset)

        ## Evaluation phase
        eval_acc = evaluate(model, test_loader, device)

        is_best = eval_acc > _best_acc
        if is_best:
            _best_acc = eval_acc

        if epoch % 5 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': _best_acc,
            }, is_best, checkpoint_dir=checkpoint_dir)

        train_accuracies.append(train_acc)
        test_accuracies.append(eval_acc)
        print(f"Epoch: {epoch+1}/{epochs}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Evaluation acc: {eval_acc:.4f}")
    
    print("Training completed.")

    if save_plots:
        # Plot accuracy
        plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(test_accuracies, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training and Test Accuracy")
        plt.savefig(f'{checkpoint_dir}/accuracy_plot.png')

        
    return _best_acc




