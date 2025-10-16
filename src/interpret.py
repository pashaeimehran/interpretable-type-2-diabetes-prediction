"""
interpret.py

This script loads a trained RETAIN model and a preprocessed patient dataset
to generate an interpretation plot. It identifies a sample patient who was
correctly predicted to have Type 2 Diabetes and visualizes the top clinical
codes that contributed to this prediction based on the model's attention weights.

Usage:
    python interpret.py \
        --data_path /path/to/processed_data.pkl \
        --model_path /path/to/trained_models/RETAIN_final.pth \
        --output_dir /path/to/plots
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Make sure to import the model definition from your models.py file
from models import RETAIN


def plot_interpretation(scores, codes, patient_idx, output_path):
    """
    Generates and saves a bar chart of the top contributing clinical codes.

    Args:
        scores (np.array): The contribution scores for each code in the sequence.
        codes (list): The list of medical codes for the patient.
        patient_idx (int): The index of the patient being interpreted.
        output_path (Path): The file path to save the generated plot.
    """
    # Identify the top 10 most influential codes by their contribution score
    top_indices = np.argsort(scores)[-10:]
    top_codes = [codes[i] for i in top_indices]
    top_scores = scores[top_indices]

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.barh(top_codes, top_scores, color='skyblue')
    plt.xlabel('Contribution Score (Importance)')
    plt.ylabel('Medical Code (ICD)')
    plt.title(f'Top 10 Clinical Codes Contributing to T2D Prediction (Patient {patient_idx})')
    plt.tight_layout()

    # Save the figure to the specified path
    plt.savefig(output_path)
    print(f"Interpretation plot saved to {output_path}")


def main(args):
    """Main function to run the RETAIN model interpretation."""
    # Ensure the input files exist
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    if not data_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("Data or model file not found. Please check paths.")

    # --- 1. Load Data and Mappings ---
    print("Loading preprocessed data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Create a reverse mapping from integer back to medical code string
    int_to_code = {v: k for k, v in data['code_to_int'].items()}
    sequences = data['sequences']
    labels = data['labels']

    # --- 2. Load the Trained RETAIN Model ---
    print(f"Loading trained model from {model_path}...")
    model = RETAIN(
        vocab_size=data['vocab_size'],
        embedding_dim=128,  # Must match the dimensions used during training
        hidden_size=128,    # Must match the dimensions used during training
        output_size=1
    )
    # Load the model onto the CPU for inference
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")

    # --- 3. Find a Suitable Patient to Interpret ---
    # We'll find the first patient who has T2D (label=1) and was correctly
    # predicted by the model (prediction > 0.5).
    sample_patient_idx = -1
    for i, (seq, label) in enumerate(zip(sequences, labels)):
        if label == 1:
            patient_seq_tensor = torch.tensor([seq], dtype=torch.long)
            with torch.no_grad():
                # Get the model's prediction and attention weights
                prediction, _, _ = model(patient_seq_tensor, return_attention=True)

            if prediction.item() > 0.5:
                sample_patient_idx = i
                break

    if sample_patient_idx == -1:
        print("Could not find a suitable sample patient to interpret.")
        return

    print(f"Interpreting patient index: {sample_patient_idx}")

    # --- 4. Calculate Contribution Scores ---
    # Get the attention weights for our chosen sample patient
    patient_sequence_tensor = torch.tensor([sequences[sample_patient_idx]], dtype=torch.long)
    with torch.no_grad():
        _, alpha_weights, beta_weights = model(patient_sequence_tensor, return_attention=True)

    # The contribution of each medical code is a product of its embedding,
    # the visit-level attention (alpha), and the code-level attention (beta).
    embedded = model.embedding(patient_sequence_tensor)
    contribution = alpha_weights.unsqueeze(2) * beta_weights * embedded
    contribution_scores = torch.sum(contribution, dim=2).squeeze(0).detach().numpy()

    # --- 5. Visualize and Save the Results ---
    patient_codes = [int_to_code.get(i, 'UNK') for i in sequences[sample_patient_idx]]

    # Prepare the output directory and filename
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f'interpretation_patient_{sample_patient_idx}.png'

    plot_interpretation(contribution_scores, patient_codes, sample_patient_idx, plot_path)


if __name__ == '__main__':
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate interpretation plots for a trained RETAIN model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the preprocessed data file (e.g., processed_data.pkl).'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the trained RETAIN model file (e.g., RETAIN_final.pth).'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./plots',
        help='Directory to save the generated interpretation plots.'
    )

    args = parser.parse_args()
    main(args)