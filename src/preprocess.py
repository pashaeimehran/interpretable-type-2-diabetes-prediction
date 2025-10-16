"""
preprocess.py

This script processes the raw MIMIC-IV dataset to create a cohort for the
task of predicting Type 2 Diabetes (T2D). It performs the following steps:
1.  Loads patient, admission, and diagnosis data.
2.  Identifies a cohort of adult patients.
3.  Defines the clinical outcome (T2D onset) based on ICD codes.
4.  Constructs longitudinal sequences of diagnoses for each patient.
5.  Creates labels (T2D or no T2D) based on the defined outcome.
6.  Encodes the sequences into integers and saves the processed data as a
    pickle file for model training.

Usage:
    python preprocess.py \
        --mimic_dir /path/to/mimiciv/2.2/ \
        --output_dir /path/to/processed_data/
"""
import argparse
import pickle
from datetime import timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main(args: argparse.Namespace) -> None:
    """
    Main function to execute the data preprocessing pipeline.

    Args:
        args (argparse.Namespace): Command-line arguments specifying input and
                                   output directories.
    """
    print("Starting data preprocessing...")

    # --- 1. Define Paths and Load Data ---
    # Construct paths to the required MIMIC-IV CSV files
    mimic_dir = Path(args.mimic_dir)
    hosp_path = mimic_dir / "hosp"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    print("Loading core patient, admission, and diagnosis data...")
    try:
        patients_df = pd.read_csv(hosp_path / 'patients.csv.gz')
        diagnoses_df = pd.read_csv(hosp_path / 'diagnoses_icd.csv.gz')
        admissions_df = pd.read_csv(hosp_path / 'admissions.csv.gz', parse_dates=['admittime'])
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the MIMIC-IV 'hosp' directory is correctly specified.")
        return

    # --- 2. Define the Patient Cohort ---
    # Filter for adult patients based on their `anchor_age`, which is the age at
    # their first hospital admission in the database.
    adult_patients = patients_df[patients_df['anchor_age'] >= 18]
    print(f"Identified {len(adult_patients)} adult patients.")

    # --- 3. Define the Clinical Outcome (T2D Diagnosis) ---
    # Identify diagnoses corresponding to Type 2 Diabetes using both ICD-9 ('250.x')
    # and ICD-10 ('E11.x') codes.
    t2d_codes = diagnoses_df['icd_code'].str.startswith('250') | diagnoses_df['icd_code'].str.startswith('E11')
    t2d_diagnoses = diagnoses_df[t2d_codes].copy()

    # Merge with admissions data to get the timestamp for each diagnosis
    t2d_diagnoses = t2d_diagnoses.merge(
        admissions_df[['subject_id', 'hadm_id', 'admittime']], on=['subject_id', 'hadm_id']
    )

    # For each patient, find the time of their very first T2D diagnosis
    first_t2d_diagnosis_time = t2d_diagnoses.loc[t2d_diagnoses.groupby('subject_id')['admittime'].idxmin()]
    first_t2d_diagnosis_time = first_t2d_diagnosis_time[['subject_id', 'admittime']].rename(
        columns={'admittime': 't2d_onset_time'}
    )
    print(f"Found {len(first_t2d_diagnosis_time)} patients with at least one T2D diagnosis.")

    # --- 4. Create Patient Sequences and Labels ---
    # Merge all diagnoses with their admission times to build patient histories
    all_diagnoses = diagnoses_df.merge(
        admissions_df[['subject_id', 'hadm_id', 'admittime']], on=['subject_id', 'hadm_id']
    )

    # Create the final labeled patient dataset by joining the adult cohort with
    # the T2D onset times. Patients without T2D will have 'NaN' in 't2d_onset_time'.
    labeled_patients = adult_patients.merge(first_t2d_diagnosis_time, on='subject_id', how='left')

    # Define a prediction window: we use patient history up to 1 year before
    # T2D onset to predict the diagnosis.
    prediction_window = timedelta(days=365)
    
    patient_sequences = []
    labels = []

    print("Generating patient sequences and labels...")
    for _, patient in tqdm(labeled_patients.iterrows(), total=labeled_patients.shape[0]):
        subject_id = patient['subject_id']
        patient_history = all_diagnoses[all_diagnoses['subject_id'] == subject_id]

        if pd.notna(patient['t2d_onset_time']):
            # This is a positive case (patient develops T2D)
            t2d_time = patient['t2d_onset_time']
            # We only use the medical history that occurred *before* the prediction window
            history_before_onset = patient_history[patient_history['admittime'] < t2d_time - prediction_window]
            
            if not history_before_onset.empty:
                patient_sequences.append(history_before_onset['icd_code'].tolist())
                labels.append(1)  # Positive label
        else:
            # This is a negative case (patient does not have a T2D diagnosis in the record)
            if not patient_history.empty:
                patient_sequences.append(patient_history['icd_code'].tolist())
                labels.append(0)  # Negative label
    
    print(f"Generated sequences for {len(patient_sequences)} patients.")

    # --- 5. Encode and Save the Processed Data ---
    # Create a vocabulary to map each unique ICD code string to an integer
    all_codes = [code for seq in patient_sequences for code in seq]
    unique_codes = sorted(list(set(all_codes)))
    # Start mapping from 1, as 0 will be reserved for padding
    code_to_int = {code: i + 1 for i, code in enumerate(unique_codes)}
    
    # Convert the sequences of code strings to sequences of integers
    sequences_as_int = [[code_to_int[code] for code in seq] for seq in patient_sequences]

    # Package all necessary data into a dictionary for easy loading during training
    output_data = {
        'sequences': sequences_as_int,
        'labels': labels,
        'code_to_int': code_to_int,
        'vocab_size': len(unique_codes) + 1  # Add 1 for the padding token (0)
    }

    # Save the final data object as a pickle file
    output_file_path = output_dir / 'processed_t2d_data.pkl'
    with open(output_file_path, 'wb') as f:
        pickle.dump(output_data, f)
        
    print(f"Preprocessing finished. Processed data saved to {output_file_path}")


if __name__ == '__main__':
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Preprocess MIMIC-IV data for Type 2 Diabetes prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--mimic_dir',
        type=str,
        required=True,
        help="Path to the root directory of the MIMIC-IV dataset (e.g., '/path/to/mimic-iv/2.2/')."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help="Directory to save the processed output file."
    )
    
    args = parser.parse_args()
    main(args)