# Predicting Type 2 Diabetes with an Interpretable Deep Learning Model

![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![Framework PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![License MIT](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the source code for an end-to-end deep learning project aimed at predicting the future onset of Type 2 Diabetes (T2D) from longitudinal Electronic Health Record (EHR) data. The core of this project is the implementation of the **RETAIN (Reverse Time Attention Model)**, an interpretable architecture that provides insight into its decision-making process.

This project was developed as a skills demonstration for the **dAIbetes – Machine Learning for Type 2 Diabetes Research (ID 2514)** position.

---

## 🎯 Key Features

* **End-to-End Pipeline:** From raw MIMIC-IV data to a trained, interpretable model.
* **Interpretable AI:** Implementation of the RETAIN model to identify key clinical risk factors driving predictions.
* **Performance Baseline:** An LSTM model is included for a direct performance comparison against a standard deep learning architecture.

---

## 🏥 Dataset

This project utilizes the **MIMIC-IV Clinical Database (v2.2)**, a large, de-identified database of patient data from a US-based hospital.

* **Cohort:** Adult patients (age ≥ 18 at first admission).
* **Task:** Predict if a patient will be diagnosed with T2D based on their prior sequence of ICD diagnosis codes.
* **Outcome:** The label is positive (1) if a patient has a recorded T2D diagnosis (ICD-9 codes starting with `250` or ICD-10 codes starting with `E11`).

> **Note:** Access to the MIMIC-IV dataset is required and can be obtained through the [PhysioNet Credentialing process]([https://physionet.org/content/mimiciv/latest/](https://physionet.org/content/mimiciv/3.1/)).
> **Note:** Access to the MIMIC-IV dataset is required and can be obtained through the [PhysioNet Credentialing process]().

---

## 📂 Project Structure

The repository is organized using a standard project structure to ensure clarity and maintainability.

```
dAIbetes_project/
│
├── .gitignore               # --- Specifies files for Git to ignore
├── README.md                # --- This project overview
├── environment.yml          # --- Conda environment for reproducibility
│
├── data/                    # --- (Ignored by Git) All project data
│   ├── raw/                 # --- Original MIMIC-IV CSVs
│   └── processed/           # --- Output of the preprocessing script
│
├── models/                  # --- (Ignored by Git) Trained model .pth files
│
├── results/                 # --- Output plots and figures
│   └── interpretation_plot.png
│
├── scripts/                 # --- Helper scripts (e.g., Slurm templates)
│   ├── run_preprocess.sbatch.template
│   └── run_training.sbatch.template
│
└── src/                     # --- All Python source code
    ├── __init__.py
    ├── preprocess.py        # --- Script for data cleaning and cohort creation
    ├── models.py            # --- PyTorch model definitions (LSTM, RETAIN)
    ├── train.py             # --- Script for model training
    ├── utils.py             # --- Data loading and helper functions
    └── interpret.py         # --- Script to generate attention-based plots
```

---

## 🛠️ Methodology

1.  **Preprocessing:** A cohort of adult patients was extracted from MIMIC-IV. For each patient, a chronological sequence of all recorded ICD diagnosis codes was constructed. Labels were assigned based on the presence of a T2D diagnosis, using a one-year washout period to ensure the model predicts a future onset.

2.  **Modeling:** Two deep learning models were implemented in PyTorch:
    * **LSTM Baseline:** A standard Long Short-Term Memory network to establish a performance baseline.
    * **RETAIN:** An interpretable model that uses two GRUs to generate attention weights at both the visit level (which visit is important?) and the code level (which diagnosis is important?). This allows us to trace the prediction back to specific clinical events.

---

## 📊 Results & Interpretation

Both models were trained for 20 epochs on the preprocessed dataset. The performance was evaluated using the Area Under the Receiver Operating Characteristic Curve (AUC-ROC).

| Model      | Final Training AUC |
| :--------- | :----------------: |
| LSTM       |       0.9931       |
| **RETAIN** |     **0.9987** |

The RETAIN model not only achieved a slightly higher AUC, demonstrating state-of-the-art predictive performance, but its primary advantage is its interpretability.

### Model Interpretation

The model's interpretability allows us to inspect the drivers of its prediction for any given patient. The plot below shows the top 10 medical codes that contributed most significantly to a positive T2D prediction for a sample patient.



**Analysis:** The model has learned clinically relevant patterns. It correctly identified well-known comorbidities and risk factors for Type 2 Diabetes, such as **Depressive Disorder (ICD code `311`)**, **Hyperlipidemia (code `2724`)**, and **Obstructive Sleep Apnea (code `32723`)**, as the most important factors for this patient's prediction. This transparency is crucial for building trust and validating the model's clinical utility.

---

## 🚀 Installation & Usage

Follow these steps to reproduce the project results.

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/dAIbetes_project.git](https://github.com/your-username/dAIbetes_project.git)
cd dAIbetes_project
```

**2. Create the Conda Environment**
```bash
conda activate daibetes-env
```

**3. Download Data**
Download the MIMIC-IV dataset from PhysioNet and place the `.csv.gz` files into the `data/raw/hosp/` directory.

**4. Run Data Preprocessing**
This script will generate the `processed_t2d_data.pkl` file in the `data/processed/` directory.
```bash
python src/preprocess.py \
    --mimic_dir ./data/raw/ \
    --output_dir ./data/processed/
```

**5. Train the Models**
Run the training script for the desired model. The trained model will be saved in the `models/` directory.
```bash
# Train the RETAIN model
python src/train.py \
    --data_path ./data/processed/processed_t2d_data.pkl \
    --model RETAIN \
    --output_dir ./models/

# Train the LSTM model
python src/train.py \
    --data_path ./data/processed/processed_t2d_data.pkl \
    --model LSTM \
    --output_dir ./models/
```

**6. Generate Interpretation Plot**
Run the interpretation script on a trained RETAIN model to generate the contribution plot.
```bash
python src/interpret.py \
    --data_path ./data/processed/processed_t2d_data.pkl \
    --model_path ./models/RETAIN_final.pth \
    --output_dir ./results/
```
