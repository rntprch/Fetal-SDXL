import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import models, transforms
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from PIL import Image


class FetalPlanesDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file, sep=";")
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df[self.df['Plane'] ==
                          'Fetal brain'].reset_index(drop=True)

        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(self.df['Brain_plane'].unique().tolist())
        self.class_to_idx = {cls_name: i for i,
                             cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image_name'] if row['Image_name'].lower().endswith(
            '.png') else row['Image_name'] + '.png'
        img_path = self.img_dir / img_name

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.class_to_idx[row['Brain_plane']], dtype=torch.long)


def get_args():
    parser = argparse.ArgumentParser(
        description="Standard Fetal Brain Classifier")
    parser.add_argument("--csv", type=str,
                        default="./FETAL_PLANES_DB_data.csv")
    parser.add_argument("--img_dir", type=str, default="./Images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = get_args()
    set_seed(args.seed)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    full_dataset = FetalPlanesDataset(
        Path(args.csv), Path(args.img_dir), transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [
                                    train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epoch_metrics = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, pred = outputs.max(1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        epoch_metrics.append(metrics)
        print(f"Epoch {epoch+1} | Acc: {acc:.4f} | F1: {f1:.4f}")

    pd.DataFrame(epoch_metrics).to_csv(
        output_path / "model_metrics.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot([m['epoch'] for m in epoch_metrics], [m['accuracy']
             for m in epoch_metrics], label='Accuracy')
    plt.plot([m['epoch'] for m in epoch_metrics], [m['f1_score']
             for m in epoch_metrics], label='F1 (Macro)')
    plt.title('Validation Performance')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(output_path / "metrics_plot.png")

    torch.save(model.state_dict(), output_path / "model.pth")
    print(f"Final metrics saved to {output_path}/model_metrics.csv")


if __name__ == "__main__":
    main()
