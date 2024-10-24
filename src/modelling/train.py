import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import glob
import argparse
import csv
from network import CNN  # Importing the model from network.py
from datasets import ImageDataset  # Importing the dataset from datasets.py


def train_model(args):
    # Load data
    train_paths = glob.glob(args.data_path + "/train/*")
    val_paths = glob.glob(args.data_path + "/val/*")
    
    train_data = ImageDataset(train_paths, num_classes=args.classes)
    val_data = ImageDataset(val_paths, num_classes=args.classes)

    print(f"Training samples: {train_data.__len__()}") 
    print(f"Validation samples: {val_data.__len__()}")

    # Prepare data loaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=val_data.__len__())

    # Initialize model
    if args.classes == 1:
        pass  # Regression task
        model = CNN(num_classes=1,input_dim=args.input_dim)  # 1 output for regression
        criterion = nn.MSELoss()
    else:
        model = CNN(num_classes=args.classes,input_dim=args.input_dim)  # 1 output for regression
        criterion = nn.CrossEntropyLoss()

    device = torch.device('mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    min_loss = np.inf
    history = []

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            images, targets = next(iter(valid_loader))
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            valid_loss = criterion(outputs, targets)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {valid_loss.item():.4f}")
        
        # Save model if validation loss improves
        if valid_loss < min_loss:
            torch.save(model, args.model_path)
            min_loss = valid_loss
            print(f"Model saved at {args.model_path}")

        # Append epoch and validation loss to history
        history.append([epoch + 1, valid_loss.item()])

    # Save training history to CSV
    csv_path = args.model_path.replace(".pth", ".csv")
    with open(csv_path, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Validation Loss"])
        writer.writerows(history)


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a CNN on images.")
    
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset folder')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--classes', type=int, default=4, help='Number of classes (1 for regression)')
    parser.add_argument('--input_dim', type=int, default=256, help='Input dimension of the image')

    args = parser.parse_args()
    
    # Train the model
    train_model(args)


if __name__ == "__main__":
    main()
