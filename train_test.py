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
sys.path.append(os.path.join(os.path.dirname(__file__), "dataset"))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

import dataset 
import utils



#################################

# 1. Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Setup seed
torch.manual_seed(42) 
torch.cuda.manual_seed(42)
random.seed(42)

# 3. Setting up hyperparameters
BATCH_SIZE = 128
EPOCHS = 75
LEARNING_RATE = 3e-4
NUM_CLASSES = 10
IMAGE_WIDTH = 32
IMAGE_LENGTH = 32
PATCH_WIDTH, PATCH_LENGTH = 8,8 
CHANNELS = 3
EMBED_DIM = 512
NUM_HEADS = 4
DEPTH = 4
MLP_DIM = 512
DROP_RATE = 0.1

# 4. Define image transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5), (0.5)),

# ])

# 5. Import dataset
path_to_dataset = '/Users/loucamalerba/Desktop/captcha_dataset_detection/Vision_transformer_digits/digits_dataset/images'

transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_LENGTH, IMAGE_WIDTH)),
    # transforms.RandomHorizontalFlip(0.3),
    # transforms.RandomVerticalFlip(0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(((0.5), (0.5), (0.5)), ((0.5), (0.5), (0.5))),
    
])

test_transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_LENGTH, IMAGE_WIDTH)),
    # transforms.RandomHorizontalFlip(0.3),
    # transforms.RandomVerticalFlip(0.3),
    # transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(((0.5), (0.5), (0.5)), ((0.5), (0.5), (0.5))),
    
])

train_dataset = dataset.ImageCaptchaDataset(root=path_to_dataset,
                                 train=True,
                                 img_h=IMAGE_LENGTH, img_w=IMAGE_WIDTH,
                                 transform=transform)


test_dataset = dataset.ImageCaptchaDataset(root=path_to_dataset,
                                 train=False,
                                 img_h=IMAGE_LENGTH, img_w=IMAGE_WIDTH,
                                 transform=test_transform)




# 6. Convert datasets to Dataloaders
## = Turn datasets into batches
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False) # Considered as a good pratice to not shuffle the test data


# 7. Import and instantiate model
from models.model import VisionTransformer   

model = VisionTransformer(
    [IMAGE_WIDTH, IMAGE_LENGTH], [PATCH_WIDTH, PATCH_LENGTH], CHANNELS, NUM_CLASSES,
    EMBED_DIM, DEPTH, NUM_HEADS, MLP_DIM, DROP_RATE
).to(device)

print(model)

# 8. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 9. Training loop
def train(model, loader, optimizer, criterion):
    # Set the mode of the model into training
    model.train()

    total_loss, correct = 0, 0

    for x, y in loader:
        # Uncomment to visualize the images if RGB is used
        # plt.figure(figsize=(5, 5))
        # print(torch.unique(x[0]))
        # print(x[0].shape)
        # print(x[0].permute(1, 2, 0).shape)
        # plt.imshow(x[0].permute(1, 2, 0).numpy())
        # plt.show()

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
    return total_loss / len(loader.dataset), correct / len(loader.dataset)
     


def evaluate(model, loader):
    model.eval() # Set the mode of the model into evlauation
    correct = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(dim=1) == y).sum().item()
    return correct / len(loader.dataset)


## Training the model
train_accuracies = []
test_accuracies = []

_best_acc = 0.0
for epoch in tqdm(range(EPOCHS)):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    test_acc = evaluate(model, test_loader)

    is_best = test_acc > _best_acc
    if is_best:
        _best_acc = test_acc

    if epoch % 5 == 0:
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': _best_acc,
        }, is_best, checkpoint_dir='checkpoints')

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    print(f"Epoch: {epoch+1}/{EPOCHS}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")
    

print("Train accuracies:", train_accuracies)
print("Test accuracies:", test_accuracies)


# Plot accuracy
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Test Accuracy")
plt.show()
     

import random

print(len(test_dataset))

print(test_dataset[0][0].unsqueeze(dim=0).shape)

print(test_dataset[0][0] / 2 + 0.5)


def predict_and_plot_grid(model,
                          dataset,
                          classes,
                          grid_size=3):
    model.eval()
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(9, 9))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = random.randint(0, len(dataset) - 1)
            img, true_label = dataset[idx]
            input_tensor = img.unsqueeze(dim=0).to(device)
            with torch.inference_mode():
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
            img = img / 2 + 0.5 # Unormalize our images to be able to plot them with matplotlib
            npimg = img.cpu().numpy()
            axes[i, j].imshow(np.transpose(npimg, (1, 2, 0)))
            truth = classes[true_label] == classes[predicted.item()]
            if truth:
                color = "g"
            else:
                color = "r"

            axes[i, j].set_title(f"Truth: {classes[true_label]}\n, Predicted: {classes[predicted.item()]}", fontsize=10, c=color)
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()
     

predict_and_plot_grid(model=model,
                      dataset=test_dataset,
                      classes=test_dataset.classes,
                      grid_size=3)


def fail_cases(model, dataset, classes):
    model.eval()
    fail_cases = []
    with torch.inference_mode():
        for i in range(len(dataset)):
            img, true_label = dataset[i]
            input_tensor = img.unsqueeze(dim=0).to(device)
            with torch.inference_mode():
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)

            if predicted.item() != true_label:
                img = img / 2 + 0.5
                npimg = img.cpu().numpy()
                plt.imshow(np.transpose(npimg, (1, 2, 0)))
                plt.title(f"Truth: {classes[true_label]}\n, Predicted: {classes[predicted.item()]}", fontsize=10)
                plt.axis("off")
                plt.show()

    return fail_cases

fail_cases(model=model,
           dataset=test_dataset,
           classes=test_dataset.classes)