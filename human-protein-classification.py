import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 25
NUM_CLASSES = 28

train_csv = pd.read_csv("dataset/train.csv")

# Convert target labels into one-hot encoding
def create_multilabel(target):
    label = np.zeros(NUM_CLASSES)
    for t in str(target).split():
        label[int(t)] = 1
    return label

train_csv['target_vec'] = train_csv['Target'].apply(create_multilabel)

train_df, valid_df = train_test_split(train_csv, test_size=0.2, random_state=42)

class ProteinDataset(Dataset):
    def __init__(self, dataframe, img_dir):
        self.df = dataframe
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['Id']

        red = cv2.imread(f"{self.img_dir}/{img_id}_red.png", cv2.IMREAD_GRAYSCALE)
        green = cv2.imread(f"{self.img_dir}/{img_id}_green.png", cv2.IMREAD_GRAYSCALE)
        blue = cv2.imread(f"{self.img_dir}/{img_id}_blue.png", cv2.IMREAD_GRAYSCALE)
        yellow = cv2.imread(f"{self.img_dir}/{img_id}_yellow.png", cv2.IMREAD_GRAYSCALE)

        red = cv2.resize(red, (IMG_SIZE, IMG_SIZE))
        green = cv2.resize(green, (IMG_SIZE, IMG_SIZE))
        blue = cv2.resize(blue, (IMG_SIZE, IMG_SIZE))
        yellow = cv2.resize(yellow, (IMG_SIZE, IMG_SIZE))

        image = np.stack([red, green, blue, yellow], axis=0)
        image = image / 255.0

        label = np.array(row['target_vec'], dtype=np.float32)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

train_dataset = ProteinDataset(train_df, "dataset/train")
valid_dataset = ProteinDataset(valid_df, "dataset/train")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CustomCNN().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_macro_f1 = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation F1 Score
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            preds = (probs > 0.3).astype(int)

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Training Loss: {running_loss / len(train_loader):.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print("-" * 50)

    # Save best model
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        torch.save(model.state_dict(), "backend/model/protein_model.pth")
        print("Best model saved!")

print("Training completed successfully")
print(f"Best Macro F1 Score: {best_macro_f1:.4f}")