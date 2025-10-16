"""
train.py

This script trains a deep learning model for Type 2 Diabetes (T2D) prediction.
It loads preprocessed patient sequence data, initializes a specified model
(LSTM or RETAIN), and runs the training loop for a given number of epochs.
The final trained model is saved to disk.

Usage:
    python train.py \
        --data_path /path/to/processed_data.pkl \
        --model RETAIN \
        --output_dir /path/to/trained_models \
        --epochs 20 \
        --learning_rate 0.001 \
        --batch_size 64
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Import model and data loading utilities from other project files
from models import RETAIN, LSTMModel
from utils import load_data


def run_training_epoch(model, dataloader, optimizer, criterion, device):
    """
    Executes a single training epoch.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        dataloader (DataLoader): The DataLoader for the training data.
        optimizer (Optimizer): The optimizer to use for training.
        criterion (Module): The loss function.
        device (str): The device to train on ('cuda' or 'cpu').

    Returns:
        tuple: A tuple containing the average loss and AUC for the epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0
    all_labels = []
    all_preds = []

    # Use tqdm for a progress bar over the batches
    for sequences, labels in tqdm(dataloader, desc="Training Epoch"):
        # Move data to the selected device (GPU or CPU)
        sequences, labels = sequences.to(device), labels.to(device).unsqueeze(1)

        # Standard training steps
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate loss and predictions for epoch-level metrics
        total_loss += loss.item()
        all_labels.extend(labels.cpu().detach().numpy())
        all_preds.extend(outputs.cpu().detach().numpy())

    avg_loss = total_loss / len(dataloader)

    # Calculate AUC, handling the edge case where a batch might have only one class
    if len(set(label.item() for label in all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        auc = float('nan')  # Not applicable if only one class is present

    return avg_loss, auc


def main(args: argparse.Namespace) -> None:
    """
    Main function to orchestrate the model training process.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    print(f"Starting training for model: {args.model}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    dataloader, vocab_size = load_data(args.data_path, args.batch_size)
    print(f"Data loaded. Vocabulary size: {vocab_size}")

    # --- 2. Initialize Model, Optimizer, and Loss Function ---
    if args.model == 'LSTM':
        model = LSTMModel(vocab_size=vocab_size, embedding_dim=128, hidden_size=128, output_size=1)
    elif args.model == 'RETAIN':
        model = RETAIN(vocab_size=vocab_size, embedding_dim=128, hidden_size=128, output_size=1)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model.to(device)

    # Adam is a good default optimizer for many deep learning tasks
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Binary Cross-Entropy is the standard loss function for binary classification
    criterion = nn.BCELoss()

    # --- 3. Training Loop ---
    print("Starting training loop...")
    for epoch in range(args.epochs):
        avg_loss, auc = run_training_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_loss:.4f} | Train AUC: {auc:.4f}")

    # --- 4. Save the Final Model ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f'{args.model}_final.pth'

    # Save the model's learned parameters
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining finished. Model saved to {model_path}")


if __name__ == '__main__':
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Train a deep learning model for T2D prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the preprocessed data file (e.g., processed_data.pkl).'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./trained_models',
        help='Directory to save the trained model.'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['LSTM', 'RETAIN'],
        required=True,
        help='The model architecture to train.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate for the optimizer.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Number of samples per training batch.'
    )

    args = parser.parse_args()
    main(args)