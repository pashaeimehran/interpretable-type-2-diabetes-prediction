"""
utils.py

This script provides utility functions and classes for handling the preprocessed
patient data for the T2D prediction task. It includes:
1.  A custom PyTorch Dataset class to wrap the patient sequences and labels.
2.  A collate function to handle padding of variable-length sequences.
3.  A data loading function to create and return a PyTorch DataLoader.

These utilities are essential for efficiently feeding data into the neural
network during training.
"""
import pickle
from pathlib import Path
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class PatientSequenceDataset(Dataset):
    """
    Custom PyTorch Dataset for patient sequences.

    This class takes a list of patient sequences and their corresponding labels
    and prepares them for use with a PyTorch DataLoader.

    Args:
        sequences (list[list[int]]): A list of patient sequences, where each
                                     sequence is a list of integer-coded
                                     medical codes.
        labels (list[int]): A list of binary labels (0 or 1) corresponding
                            to each patient sequence.
    """
    def __init__(self, sequences: List[List[int]], labels: List[int]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        """Returns the total number of patients in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single patient's data at the specified index.

        Args:
            idx (int): The index of the patient to retrieve.

        Returns:
            tuple: A tuple containing the patient's sequence and label as
                   PyTorch tensors.
        """
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return sequence, label


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for the DataLoader.

    This function takes a list of samples (each being a sequence-label pair)
    and pads the sequences to the same length within the batch. This is
    necessary because RNNs require fixed-size input tensors.

    Args:
        batch (list): A list of tuples, where each tuple contains a sequence
                      tensor and a label tensor.

    Returns:
        tuple: A tuple containing a padded sequence tensor and a stacked
               label tensor for the entire batch.
    """
    sequences, labels = zip(*batch)

    # `pad_sequence` stacks the sequences and pads them to the length of the
    # longest sequence in the batch. `batch_first=True` ensures the output
    # shape is (batch_size, seq_length).
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Stack the individual label tensors into a single tensor
    stacked_labels = torch.stack(labels)
    
    return padded_sequences, stacked_labels


def load_data(data_path: str, batch_size: int) -> Tuple[DataLoader, int]:
    """
    Loads the processed data and creates a DataLoader.

    Args:
        data_path (str): The path to the processed data pickle file.
        batch_size (int): The number of samples per batch.

    Returns:
        tuple: A tuple containing the configured DataLoader and the vocabulary size.
    """
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Create an instance of our custom dataset
    dataset = PatientSequenceDataset(data['sequences'], data['labels'])

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    return dataloader, data['vocab_size']


# This block allows you to run the script directly to test its functionality
if __name__ == '__main__':
    # Create a dummy data file for testing purposes
    print("Creating a dummy data file for testing...")
    dummy_data = {
        'sequences': [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
        'labels': [0, 1, 0],
        'code_to_int': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9},
        'vocab_size': 10
    }
    dummy_path = Path('./dummy_processed_data.pkl')
    with open(dummy_path, 'wb') as f:
        pickle.dump(dummy_data, f)
    
    print(f"Dummy data saved to {dummy_path}")

    # Test the load_data function
    print("\nTesting load_data function...")
    try:
        test_dataloader, test_vocab_size = load_data(str(dummy_path), batch_size=2)
        print(f"Successfully created DataLoader. Vocabulary size: {test_vocab_size}")
        
        # Check the output of one batch
        for sequences_batch, labels_batch in test_dataloader:
            print("\nSample batch:")
            print(f"  Sequences shape: {sequences_batch.shape}")
            print(f"  Sequences tensor:\n{sequences_batch}")
            print(f"  Labels shape: {labels_batch.shape}")
            print(f"  Labels tensor:\n{labels_batch}")
            break  # Only inspect the first batch
    except Exception as e:
        print(f"An error occurred during testing: {e}")
    finally:
        # Clean up the dummy file
        if dummy_path.exists():
            dummy_path.unlink()
            print(f"\nCleaned up {dummy_path}")