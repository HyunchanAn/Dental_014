import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from dental_014.unet_model import UNet
from dental_014.unet_dataset import MandibleDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check data paths
    data_dir = os.path.join(os.path.dirname(__file__), "../data/mendeley_dataset")
    image_dir = os.path.join(data_dir, "Images")
    mask_dir = os.path.join(data_dir, "Segmentation1")
    if not os.path.exists(mask_dir):
        mask_dir = os.path.join(data_dir, "Segmentation2")

    print(f"Images: {image_dir}, Masks: {mask_dir}")

    dataset = MandibleDataset(image_dir, mask_dir, img_size=(256, 512))

    # Split train and val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    model = UNet(in_channels=3, out_channels=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 30
    best_loss = float("inf")

    print(
        f"Training U-Net on {len(train_dataset)} images, validating on {len(val_dataset)}..."
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, masks in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"
        ):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"
            ):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), "../models/unet_mandible_best.pth")
            print("Saved new best model!")


if __name__ == "__main__":
    main()
