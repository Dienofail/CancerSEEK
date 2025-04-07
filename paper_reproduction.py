import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn
import scipy
import os
import argparse
import process_supplemental_data as supp_processor
import models
import plotting
from tqdm.auto import tqdm  # Import tqdm.auto for progress bars that work in both CLI and Jupyter

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Reproduce CancerSEEK paper analysis with customizable feature transformations')
    
    # Add arguments for feature transformation
    parser.add_argument('--standardize', action='store_true', 
                        help='Standardize features using StandardScaler')
    parser.add_argument('--no-standardize', dest='standardize', action='store_false',
                        help='Do not standardize features')
    parser.add_argument('--log-transform', type=str, choices=['none', 'log2', 'log10'], default='none',
                        help='Apply log transformation to protein features (default: none)')
    parser.add_argument('--outer-splits', type=int, default=10,
                        help='Number of outer CV splits (default: 10)')
    parser.add_argument('--inner-splits', type=int, default=4,
                        help='Number of inner CV splits (default: 4)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--detection-model', type=str, choices=['LR', 'XGB', 'TF', 'MOE'], default='LR',
                        help='Model type for cancer detection (LR=Logistic Regression, XGB=XGBoost, TF=TensorFlow Late Fusion, MOE=Mixture of Experts) (default: LR)')
    parser.add_argument('--localization-model', type=str, choices=['RF', 'XGB', 'TF', 'MOE'], default='RF',
                        help='Model type for tissue localization (RF=Random Forest, XGB=XGBoost, TF=TensorFlow Fusion, MOE=Mixture of Experts) (default: RF)')
    parser.add_argument('--target-spec', type=int, default=99,
                        help='Target specificity as integer (99=0.99, 985=0.985, etc.) for evaluation (default: 985)')

    # Set defaults
    parser.set_defaults(standardize=False)

    return parser.parse_args()

# Get command line arguments
args = parse_arguments()

# Map 'none' to None for log_transform
log_transform = None if args.log_transform == 'none' else args.log_transform

# Convert target_specificity from integer to decimal (e.g., 99 -> 0.99, 985 -> 0.985)
if args.target_spec < 100:
    # Two-digit input (e.g., 99 -> 0.99)
    target_specificity = args.target_spec / 100.0
else:
    # Three-digit input (e.g., 985 -> 0.985)
    target_specificity = args.target_spec / 1000.0

print(f"Using target specificity: {target_specificity:.3f}")

# New cells to fit models and generate plots
print("Loading supplemental data...")
# Load the supplemental data
supp_data_dict = supp_processor.load_all_supplemental_files()

print("Processing patient information...")
# Create a DataFrame with patient information
clinical_df = supp_data_dict['s4_clinical_characteristics']
mutations_df = supp_data_dict['s5_mutations']
protein_df = supp_data_dict['s6_protein_conc']

# Debug: Print column names to identify structure
print("\nDebug - Clinical df columns:", clinical_df.columns.tolist())
print("\nDebug - Mutations df columns:", mutations_df.columns.tolist()[:10], "...")
print("\nDebug - Protein df columns:", protein_df.columns.tolist()[:10], "...")

# Create binary target: cancer vs. normal
clinical_df['is_cancer'] = (clinical_df['Tumor_type'] != 'Normal').astype(int)

# Fix Patient_ID format discrepancies - strip special characters like daggers (†)
clinical_df['Patient_ID'] = clinical_df['Patient_ID'].str.replace(r'[^\w\s_-]', '', regex=True)
protein_df['Patient_ID'] = protein_df['Patient_ID'].str.replace(r'[^\w\s_-]', '', regex=True)
mutations_df['Patient_ID'] = mutations_df['Patient_ID'].str.replace(r'[^\w\s_-]', '', regex=True)

print("Extracting protein measurements...")
# Extract only protein measurements from protein_df
protein_columns = [col for col in protein_df.columns if col not in ['Patient_ID', 'Sample_ID', 'Tumor_type', 'AJCC_Stage', 'CancerSEEK_Logistic_Regression_Score', 'CancerSEEK_Test_Result']]
print(f"\nDebug - Number of protein columns: {len(protein_columns)}")
print(f"Debug - First few protein columns: {protein_columns[:5]}")
protein_measurements = protein_df[['Patient_ID', 'Sample_ID'] + protein_columns].copy()

# Merge protein data with clinical data
data_df = clinical_df.merge(protein_measurements, on=['Patient_ID', 'Sample_ID'], how='left')
print(f"\nDebug - Shape after merging protein data: {data_df.shape}")
print(f"Debug - Data df columns after protein merge: {data_df.columns.tolist()[:10]}", "...")

print("Checking for omega_score...")
# Check for missing omega_score and add it from mutations_df if available
if 'omega_score' not in data_df.columns:
    # Extract relevant columns from mutations_df
    omega_cols = ['Patient_ID', 'Sample_ID']
    if 'omega_score' in mutations_df.columns:
        omega_cols.append('omega_score')
        omega_df = mutations_df[omega_cols].drop_duplicates()
        print(f"\nDebug - Found omega_score in mutations_df. Shape: {omega_df.shape}")
        
        # Merge with data_df
        data_df = data_df.merge(omega_df, on=['Patient_ID', 'Sample_ID'], how='left')
        # Fill missing omega_score with 0 (no mutation detected)
        data_df['omega_score'] = data_df['omega_score'].fillna(0)
    else:
        print("\nDebug - omega_score not found in mutations_df. Will proceed without it.")

print(f"\nDebug - Final data shape: {data_df.shape}")
print(f"Debug - First few columns: {data_df.columns.tolist()[:10]}")

print("Handling missing values...")
# Handle any remaining NAs in protein columns by imputation
# Filter to get only normal samples
normal_samples = data_df[data_df['Tumor_type'] == 'Normal']

# Impute protein values using mean of normal samples, accounting for NaN values
for col in protein_columns:
    if col in data_df.columns and data_df[col].isna().any():
        # Calculate mean of non-NaN values in normal samples
        normal_mean = normal_samples[col].dropna().mean()
        # If all normal samples have NaN for this column, use overall non-NaN mean
        if pd.isna(normal_mean):
            normal_mean = data_df[col].dropna().mean()
        data_df[col] = data_df[col].fillna(normal_mean)

# Impute omega_score if it exists and has NAs
if 'omega_score' in data_df.columns and data_df['omega_score'].isna().any():
    # Calculate mean of non-NaN omega values in normal samples
    normal_omega_mean = normal_samples['omega_score'].dropna().mean()
    # If all normal samples have NaN for omega_score, use overall non-NaN mean
    if pd.isna(normal_omega_mean):
        normal_omega_mean = data_df['omega_score'].dropna().mean()
    data_df['omega_score'] = data_df['omega_score'].fillna(normal_omega_mean)

# Create a list of columns to keep for modeling
# These include identification columns needed by the model
id_columns = ['Patient_ID', 'Sample_ID', 'Tumor_type', 'AJCC_Stage']

# Identify which columns should be treated as features
feature_columns = []
for col in data_df.columns:
    # Skip ID/metadata columns and string columns
    if (col not in id_columns and 
        col not in ['Primary_tumor_sample_ID', 'Histopathology', 'Race', 'Sex', 
                    'Plasma_volume_(mL)', 'Plasma_DNA_concentration', 
                    'CancerSEEK_Logistic_Regression_Score', 'CancerSEEK_Test_Result', 
                    'is_cancer'] and
        pd.api.types.is_numeric_dtype(data_df[col])):
        feature_columns.append(col)

print(f"\nDebug - Feature columns: {feature_columns[:5]}", "...")
print(f"Debug - Number of feature columns: {len(feature_columns)}")

# Create X with both ID columns and feature columns
X = data_df[id_columns + feature_columns].copy()
y_cancer_status = data_df['is_cancer']
y_cancer_type = data_df['Tumor_type']

print(f"\nDebug - X shape: {X.shape}")
print(f"Debug - X columns: {X.columns.tolist()[:10]}", "...")

# Define the protein features for LR model based on the paper
lr_proteins = ['CA-125', 'CA19-9', 'CEA', 'HGF', 'Myeloperoxidase', 'OPN', 'Prolactin', 'TIMP-1']

# Verify all LR proteins exist and are numeric
for protein in lr_proteins:
    if protein not in X.columns:
        print(f"Warning: Protein {protein} not found in the dataset")
    elif not pd.api.types.is_numeric_dtype(X[protein]):
        print(f"Warning: Protein {protein} is not numeric, converting...")
        X[protein] = pd.to_numeric(X[protein], errors='coerce')
        X[protein] = X[protein].fillna(X[protein].median())

# Create plots directory if it doesn't exist
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created directory: {plots_dir}")

print("\n==== Starting model training and evaluation ====")
print(f"Training combined cancer detection ({args.detection_model}) and localization ({args.localization_model}) model with {X.shape[0]} samples")
print(f"Using {len(lr_proteins)} protein features for detection model")
print(f"Performing {args.outer_splits}-fold outer CV and {args.inner_splits}-fold inner CV")
print(f"Standardization: {args.standardize}, Log transformation: {args.log_transform}")

# Fit the combined model
combined_results = models.combined_cancer_detection_and_localization(
    X, y_cancer_status, y_cancer_type, clinical_df=clinical_df,
    outer_splits=args.outer_splits, inner_splits=args.inner_splits, random_state=args.random_seed,
    standardize_features=args.standardize, log_transform=log_transform,
    detection_model=args.detection_model, localization_model=args.localization_model
)

print("\n==== Model training complete ====")
print("Generating plots...")

# Add debugging statements to check the structure of localization_results
print("\n==== Debugging localization_results ====")
print(f"Type of combined_results: {type(combined_results)}")
if 'localization_results' in combined_results:
    print(f"Type of localization_results: {type(combined_results['localization_results'])}")
    if combined_results['localization_results'] is not None:
        print(f"Keys in localization_results: {combined_results['localization_results'].keys()}")
        print(f"Type of predictions: {type(combined_results['localization_results']['predictions'])}")
        print(f"Shape of predictions: {combined_results['localization_results']['predictions'].shape}")
        if isinstance(combined_results['localization_results']['predictions'], np.ndarray):
            print("predictions is a numpy array, not a DataFrame/Series with an index")
else:
    print("localization_results key not found in combined_results")

# Get model types from results
detection_model = combined_results.get('detection_model', args.detection_model)
localization_model = combined_results.get('localization_model', args.localization_model)

# Plot ROC curve for the detection model
fig_roc, ax_roc, spec_percent = plotting.plot_roc_curve(
    y_cancer_status, 
    combined_results['detection_results']['probabilities'],
    title=f"ROC Curve for CancerSEEK ({detection_model})",
    model_type=detection_model,
    save_data=os.path.join(plots_dir, f"roc_curve_data_{detection_model}.pkl"),
    target_specificity=target_specificity
)
plt.tight_layout()
# Save ROC curve to PDF
roc_filename = f"roc_curve_{detection_model}_{spec_percent}.pdf"
plt.savefig(os.path.join(plots_dir, roc_filename), bbox_inches='tight')
print(f"Saved ROC curve to {os.path.join(plots_dir, roc_filename)}")
print(f"Saved ROC curve data to {os.path.join(plots_dir, f'roc_curve_data_{detection_model}_{spec_percent}.pkl')}")

# Calculate sensitivity at target specificity for all cancers and by subtype
sensitivity_by_subtype = plotting.calculate_sensitivity_by_subtype(
    y_cancer_status,
    combined_results['detection_results']['probabilities'],
    data_df['Tumor_type'],
    target_specificity=target_specificity
)

# Plot sensitivity by cancer type
fig_subtype, ax_subtype, spec_percent = plotting.plot_sensitivity_by_subtype(
    sensitivity_by_subtype,
    figsize=(12, 6),
    model_type=detection_model
)
plt.tight_layout()
# Save sensitivity by subtype to PDF
subtype_filename = f"sensitivity_by_subtype_{detection_model}_{spec_percent}.pdf"
plt.savefig(os.path.join(plots_dir, subtype_filename), bbox_inches='tight')
print(f"Saved sensitivity by subtype to {os.path.join(plots_dir, subtype_filename)}")

# Calculate sensitivity by cancer stage
sensitivity_by_stage = plotting.calculate_sensitivity_by_stage(
    y_cancer_status,
    combined_results['detection_results']['probabilities'],
    data_df['AJCC_Stage'],
    target_specificity=target_specificity
)

# Plot sensitivity by cancer stage
fig_stage, ax_stage, spec_percent = plotting.plot_sensitivity_by_stage(
    sensitivity_by_stage,
    figsize=(10, 6),
    model_type=detection_model
)
plt.tight_layout()
# Save sensitivity by stage to PDF
stage_filename = f"sensitivity_by_stage_{detection_model}_{spec_percent}.pdf"
plt.savefig(os.path.join(plots_dir, stage_filename), bbox_inches='tight')
print(f"Saved sensitivity by stage to {os.path.join(plots_dir, stage_filename)}")

# Generate performance summary CSV
performance_csv_path = os.path.join(plots_dir, "performance_summary.csv")
summary_df = plotting.create_performance_summary_csv(
    sensitivity_by_stage,
    performance_csv_path,
    model_type=detection_model
)
print(f"Saved performance summary to {performance_csv_path.replace('.csv', f'_{detection_model}_{spec_percent}.csv')}")

# Plot tissue localization accuracy (Fig 3 in the paper)
print("Processing tissue localization results...")
# We need to filter to only positive samples identified by CancerSEEK
# First, get threshold at target specificity
sensitivity_result = plotting.calculate_sensitivity_at_specificity(
    y_cancer_status,
    combined_results['detection_results']['probabilities'],
    target_specificity=target_specificity
)
threshold = sensitivity_result['threshold']

# Identify positive samples according to CancerSEEK threshold
positive_indices = np.where(
    (combined_results['detection_results']['probabilities'] >= threshold) & 
    (y_cancer_status == 1)
)[0]

# Filter data for tissue localization plot
y_true_filtered = [y_cancer_type.iloc[i] for i in positive_indices]

# Extract cancer types (excluding 'Normal')
cancer_types = [t for t in y_cancer_type.unique() if t != 'Normal']

# Check if we have any cancer types to ensure we don't divide by zero
if len(cancer_types) == 0:
    print("Warning: No cancer types found for localization plotting")
    cancer_types = ["Unknown"]  # Add a default to prevent errors

# If we have probabilities from the localization model
if combined_results['localization_results'] is not None:
    # Add debug info to understand data structures
    print("\n==== Debug: Tissue localization data ====")
    print(f"Number of positive indices: {len(positive_indices)}")
    print(f"Max positive index: {int(np.max(positive_indices)) if len(positive_indices) > 0 else 'None'}")
    
    predictions = combined_results['localization_results']['predictions']
    print(f"Predictions array length: {len(predictions)}")
    print(f"Predictions array type: {type(predictions)}")
    print(f"First few values in predictions: {predictions[:5] if len(predictions) >= 5 else predictions}")
    
    # Extract predictions for positive samples
    y_pred_proba = {}
    
    for i, idx in enumerate(positive_indices):
        # Make sure idx is valid by using modulo or checking array bounds
        idx_int = int(idx)  # Convert numpy int to Python int
        if idx_int >= len(predictions):
            continue  # Skip this index if it's out of bounds
            
        # Create probability dict for this sample
        proba_dict = {}
        for cancer_type in cancer_types:
            # Get "predicted probabilities" for each cancer type (simplified approach)
            pred_type = predictions[idx_int] if len(predictions) > idx_int else ""
            if cancer_type == pred_type and pred_type != "":
                proba_dict[cancer_type] = 0.9  # High probability for predicted class
            else:
                proba_dict[cancer_type] = 0.1 / (len(cancer_types) - 1) if len(cancer_types) > 1 else 0.1  # Distribute remaining probability
        
        y_pred_proba[i] = proba_dict

    # Plot tissue localization accuracy
    fig_loc, ax_loc = plotting.plot_tissue_localization_accuracy(
        y_true_filtered,
        y_pred_proba,
        cancer_types=cancer_types,
        figsize=(12, 8),
        detection_model=detection_model,
        localization_model=localization_model
    )
    # No need to update the title here as it's now handled in the plotting function
    plt.tight_layout()
    # Save tissue localization accuracy to PDF
    loc_filename = f"tissue_localization_accuracy_{detection_model}_{localization_model}.pdf"
    plt.savefig(os.path.join(plots_dir, loc_filename), bbox_inches='tight')
    print(f"Saved tissue localization accuracy to {os.path.join(plots_dir, loc_filename)}")
else:
    print("No localization results available.")

print("\n==== All processing complete ====")