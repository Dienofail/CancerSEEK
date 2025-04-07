import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import pickle
from tqdm.auto import tqdm
import process_supplemental_data as supp_processor
import models
import plotting

def generate_synthetic_feature(X, y, alpha, orthogonality=1.0, max_iter=100, tol=1e-3):
    """
    Generates a synthetic feature that has a target correlation with y
    and is linearly uncorrelated with the features in X to a tunable degree.

    Args:
        X (pd.DataFrame or np.ndarray): DataFrame or array of existing features (n_samples, n_features).
        y (pd.Series or np.ndarray): Target variable (n_samples,).
        alpha (float): Target correlation between the synthetic feature and y.
        orthogonality (float, optional): Degree of orthogonalization with existing features (0.0 to 1.0).
            - 0.0: No orthogonalization (keeps correlations with X)
            - 1.0: Full orthogonalization (completely uncorrelated with X)
            - Values in between: Partial orthogonalization
        max_iter (int, optional): Maximum iterations for refining initial correlation. Defaults to 100.
        tol (float, optional): Tolerance for matching the target correlation alpha. Defaults to 1e-3.

    Returns:
        np.ndarray: The generated synthetic feature (n_samples,).
                    Returns None if the process fails (e.g., cannot achieve initial correlation).
    """
    # Input validation for orthogonality parameter
    if not (0 <= orthogonality <= 1):
        raise ValueError("orthogonality must be between 0 and 1")
        
    # ---------------------------------------------------
    # 1. INITIALIZE
    # ---------------------------------------------------
    X_input_df = None
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist() # Keep track of names
        X_input_df = X # Keep the DataFrame for easier handling later
        X_values = X.values
    elif isinstance(X, np.ndarray):
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_values = X
    else:
        raise TypeError("X must be a pandas DataFrame or a numpy array.")

    if isinstance(y, pd.Series):
        y = y.values

    n_samples, n_features = X_values.shape

    if y.shape[0] != n_samples:
        raise ValueError("X and y must have the same number of samples.")
    if not (-1 <= alpha <= 1):
        raise ValueError("alpha must be between -1 and 1.")

    # Center X columns
    X_mean = np.mean(X_values, axis=0)
    X_centered = X_values - X_mean

    # Use y directly (as per pseudocode skipping standardization)
    y_target = y

    # ---------------------------------------------------
    # 2. CREATE AN INITIAL Z_raw THAT CORRELATES WITH y
    # ---------------------------------------------------
    a = 1.0
    y_std_val = np.std(y_target)
    if y_std_val < 1e-8:
         print("Warning: Target variable y has near-zero variance.")
         noise = np.random.normal(loc=0, scale=1, size=n_samples)
         # Even for zero variance y, we still apply the requested orthogonality
         if orthogonality > 0:
             lin_reg = LinearRegression()
             lin_reg.fit(X_centered, noise)
             noise_hat = lin_reg.predict(X_centered)
             Z_uncorr_full = noise - noise_hat
             Z_uncorr = (orthogonality * Z_uncorr_full) + ((1 - orthogonality) * noise)
         else:
             Z_uncorr = noise
         return Z_uncorr

    noise_level = y_std_val
    iteration = 0
    best_corr_diff = float('inf')
    best_Z_raw = None

    # Suppress initial generation print message if not needed for clarity in loops
    # print(f"Attempting to generate initial feature with target correlation {alpha:.4f} with y...")
    while iteration < max_iter:
        eps = np.random.normal(loc=0, scale=noise_level, size=n_samples)
        Z_raw_candidate = a * y_target + eps
        Z_raw_std = np.std(Z_raw_candidate)
        if Z_raw_std < 1e-8:
             corr_candidate = 0.0
        else:
             corr_matrix = np.corrcoef(Z_raw_candidate, y_target)
             corr_candidate = corr_matrix[0, 1]
             if np.isnan(corr_candidate):
                 corr_candidate = 0.0

        corr_diff = abs(corr_candidate - alpha)
        if corr_diff < best_corr_diff:
            best_corr_diff = corr_diff
            best_Z_raw = Z_raw_candidate.copy()

        adjustment_factor = 1.05
        if corr_candidate > alpha:
            a *= (1 / adjustment_factor)
            noise_level *= adjustment_factor
        else:
            a *= adjustment_factor
            noise_level *= (1 / adjustment_factor)
            noise_level = max(noise_level, 1e-6)

        if best_corr_diff < tol:
            # print(f"  Reached target correlation {alpha:.4f} (actual: {corr_candidate:.4f}) in {iteration + 1} iterations.")
            break
        iteration += 1

    if best_Z_raw is None:
        print(f"Warning: Failed to generate initial Z_raw with correlation near {alpha} after {max_iter} iterations. Best diff: {best_corr_diff:.4f}")
        return None

    Z_raw = best_Z_raw
    # initial_corr = np.corrcoef(Z_raw, y_target)[0, 1]
    # print(f"  Final initial correlation with y: {initial_corr:.4f}")

    # ---------------------------------------------------
    # 3. REMOVE CORRELATION WITH EXISTING FEATURES (TUNABLE)
    # ---------------------------------------------------
    if orthogonality > 0:  # Skip if orthogonality=0 (no orthogonalization)
        # Compute fully orthogonalized version
        lin_reg = LinearRegression()
        lin_reg.fit(X_centered, Z_raw)
        Z_hat = lin_reg.predict(X_centered)
        Z_uncorr_full = Z_raw - Z_hat
        
        # Check if the fully orthogonalized feature is essentially constant
        if np.std(Z_uncorr_full) < 1e-8:
            print(f"Warning: Full orthogonalization resulted in a near-constant feature. All variance explained by X.")
            if orthogonality == 1.0:
                # If full orthogonality was requested, return the constant feature
                return Z_uncorr_full
            else:
                # If partial orthogonality was requested, we can still proceed
                print(f"Continuing with partial orthogonalization (orthogonality={orthogonality})")
        
        # Apply the specified degree of orthogonality
        # For orthogonality = 0, Z_uncorr = Z_raw (no orthogonalization)
        # For orthogonality = 1, Z_uncorr = Z_uncorr_full (full orthogonalization)
        # For values in between, it's a weighted average
        Z_uncorr = (orthogonality * Z_uncorr_full) + ((1 - orthogonality) * Z_raw)
    else:
        # Skip orthogonalization entirely
        Z_uncorr = Z_raw
        print(f"Skipping orthogonalization as requested (orthogonality=0)")

    # ---------------------------------------------------
    # 4. RESCALE TO RESTORE CORRELATION ~ alpha
    # ---------------------------------------------------
    # print("Rescaling orthogonalized feature to restore target correlation with y...")
    corr_matrix_uncorr = np.corrcoef(Z_uncorr, y_target)
    current_corr = corr_matrix_uncorr[0, 1]

    if np.isnan(current_corr):
        print("Warning: Correlation calculation failed after orthogonalization (NaN). Returning unscaled feature.")
        return Z_uncorr

    if abs(current_corr) < 1e-8:
        print(f"Warning: Correlation with y became near zero ({current_corr:.2e}) after orthogonalization. Cannot rescale to target alpha {alpha}. Returning unscaled feature.")
        Z_final = Z_uncorr
    else:
        scale_factor = alpha / current_corr
        Z_final = Z_uncorr * scale_factor
        # print(f"  Rescaled using factor: {scale_factor:.4f}")

    # ---------------------------------------------------
    # 5. OUTPUT & CHECK (Only activate for debugging if needed)
    # ---------------------------------------------------
    # final_corr_with_y = np.corrcoef(Z_final, y_target)[0, 1]
    # print(f"Final correlation with y: {final_corr_with_y:.4f} (Target: {alpha:.4f})")
    
    # if orthogonality > 0:
    #     print(f"Checking correlation with X features (orthogonality={orthogonality}):")
    #     corrs_with_X = []
    #     for j in range(n_features):
    #         corr_Xj = np.corrcoef(Z_final, X_centered[:, j])[0, 1]
    #         corrs_with_X.append(corr_Xj)
    #     max_abs_corr_X = np.max(np.abs(corrs_with_X))
    #     print(f"  Max absolute correlation with any X feature: {max_abs_corr_X:.4f}")
    #     # For orthogonality=1.0, max_abs_corr_X should be near 0
    #     # For orthogonality=0.0, max_abs_corr_X could be much higher

    return Z_final

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train logistic regression models adding synthetic orthogonal features')

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
    parser.add_argument('--target-specificity', type=float, default=0.99,
                        help='Target specificity for sensitivity calculations (default: 0.99)')
    parser.add_argument('--num-synthetic-features', type=int, default=20,
                        help='Number of synthetic orthogonal features to add (default: 20)')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Target correlation for synthetic features with the target variable y (default: 0.3)')
    parser.add_argument('--orthogonality', type=float, default=1.0,
                        help='Degree of orthogonalization with existing features (0.0 to 1.0, default: 1.0)')

    # Set defaults
    parser.set_defaults(standardize=False)

    return parser.parse_args()

def train_models_with_synthetic_features(X_orig, y, base_features, protein_features, omega_feature,
                                         clinical_df,
                                         num_synthetic_features=10, alpha=0.3, orthogonality=1.0,
                                         outer_splits=10, inner_splits=4,
                                         random_state=42, standardize_features=False,
                                         log_transform=None, target_specificity=0.99):
    """
    Train models iteratively, adding one synthetic orthogonal feature at a time.

    Parameters:
    -----------
    X_orig : pandas.DataFrame
        Original DataFrame containing all potential features for the model.
    y : pandas.Series or numpy.ndarray
        Target variable (cancer vs normal).
    base_features : list
        List of feature names to start with (e.g., original proteins + omega).
    protein_features : list
        List of protein feature names within base_features (for potential transforms).
    omega_feature : str or None
        Name of the omega score feature, if used.
    clinical_df : pandas.DataFrame
        DataFrame with clinical information.
    num_synthetic_features : int, optional
        Total number of synthetic features to add iteratively (default: 10).
    alpha : float, optional
        Target correlation for synthetic features with y (default: 0.3).
    orthogonality : float, optional
        Degree of orthogonalization with existing features (0.0 to 1.0, default: 1.0).
        - 0.0: No orthogonalization (keeps correlations with X)
        - 1.0: Full orthogonalization (completely uncorrelated with X)
        - Values in between: Partial orthogonalization
    outer_splits : int, optional
        Number of folds for outer cross-validation (default: 10).
    inner_splits : int, optional
        Number of folds for inner cross-validation (default: 4).
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    standardize_features : bool, optional
        Whether to standardize features within CV (default: False).
    log_transform : str, optional
        Type of log transformation to apply to protein features ('log2', 'log10', or None).
    target_specificity : float, optional
        Target specificity for sensitivity calculations (default: 0.99).

    Returns:
    --------
    dict
        Dictionary containing results for models with different numbers of synthetic features.
    """
    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}")

    # Create specific output directory including orthogonality parameter
    output_dir = os.path.join(plots_dir, f"synthetic_feature_ortho{orthogonality:.1f}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Initialize containers for results
    results = {}
    roc_data_files = []
    sensitivities = []
    X_current_iter = X_orig.copy()
    current_features = base_features.copy()
    added_synthetic_features = []

    # --- Train Model with 0 Synthetic Features (Base Model) --- #
    print(f"\n==== Training base model with {len(current_features)} features ====")
    from sklearn.linear_model import LogisticRegression # Moved import here
    custom_lr_model_base = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        random_state=random_state,
        max_iter=3000
    )
    model_results_base = models.nested_cross_validation(
        X_current_iter[current_features + id_columns], # Pass only relevant columns + IDs
        y,
        protein_features=protein_features,
        omega_score_col=omega_feature,
        clinical_df=clinical_df,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        random_state=random_state,
        standardize_features=standardize_features,
        log_transform=log_transform,
        model_type='LR',
        custom_model=custom_lr_model_base
    )
    results[0] = model_results_base # Store result for 0 synthetic features

    # Save ROC data for base model
    roc_data_path_base = os.path.join(output_dir, f"roc_curve_data_LR_0_synth_features.pkl")
    fig_roc_base, ax_roc_base = plotting.plot_roc_curve(
        y,
        model_results_base['probabilities'],
        title=f"ROC Curve for CancerSEEK LR with Base Features (0 Synthetic)",
        model_type=f"LR-0synth",
        save_data=roc_data_path_base
    )
    plt.tight_layout()
    roc_filename_base = os.path.join(output_dir, f"roc_curve_LR_0_synth_features.pdf")
    plt.savefig(roc_filename_base, bbox_inches='tight')
    plt.close()
    roc_data_files.append(roc_data_path_base)

    # Calculate sensitivity for base model
    sensitivity_result_base = plotting.calculate_sensitivity_at_specificity(
        y,
        model_results_base['probabilities'],
        target_specificity=target_specificity
    )
    sensitivities.append({
        'num_synthetic': 0,
        'n_features_total': len(current_features),
        'sensitivity': sensitivity_result_base['sensitivity'],
        'ci_low': sensitivity_result_base['ci_low'],
        'ci_high': sensitivity_result_base['ci_high'],
        'features': current_features.copy()
    })

    # --- Iteratively Add Synthetic Features --- #
    for k in tqdm(range(num_synthetic_features), desc="Adding synthetic features"):
        num_added = k + 1
        print(f"\n==== Generating and Training with {num_added} Synthetic Feature(s) ====")

        # Generate the next synthetic feature, orthogonal to the *current* set
        print(f"Generating synthetic feature {num_added} (alpha={alpha}, orthogonality={orthogonality})...")
        X_for_ortho = X_current_iter[current_features].copy()
        synthetic_feat_k = generate_synthetic_feature(X_for_ortho, y, alpha=alpha, orthogonality=orthogonality)

        if synthetic_feat_k is None:
            print(f"Failed to generate synthetic feature {num_added}. Stopping experiment.")
            break

        # Add the new feature to the DataFrame and feature list
        synth_feat_name = f"synthetic_{num_added}"
        X_current_iter[synth_feat_name] = synthetic_feat_k
        current_features.append(synth_feat_name)
        added_synthetic_features.append(synth_feat_name)
        print(f"Added feature: {synth_feat_name}. Total features: {len(current_features)}")

        # Create a combined feature list with original proteins + synthetic features
        combined_features = protein_features.copy() + added_synthetic_features

        # Train model with nested cross-validation using the updated feature set
        # Need to create a new model instance for each iteration potentially
        custom_lr_model_k = LogisticRegression(
             penalty='elasticnet',
             solver='saga',
             random_state=random_state,
             max_iter=3000
        )
        # Important: Pass X with only the *current* features + ID columns to nested_cv
        model_results_k = models.nested_cross_validation(
            X_current_iter[current_features + id_columns],
            y,
            protein_features=combined_features,  # Now includes synthetic features
            omega_score_col=omega_feature,
            clinical_df=clinical_df,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            random_state=random_state,
            standardize_features=standardize_features,
            log_transform=log_transform,
            model_type='LR',
            custom_model=custom_lr_model_k
        )

        # Store results
        results[num_added] = model_results_k

        # Save individual ROC curve and data
        roc_data_path = os.path.join(output_dir, f"roc_curve_data_LR_{num_added}_synth_features.pkl")
        fig_roc, ax_roc = plotting.plot_roc_curve(
            y,
            model_results_k['probabilities'],
            title=f"ROC Curve for CancerSEEK LR with {num_added} Synthetic Features",
            model_type=f"LR-{num_added}synth",
            save_data=roc_data_path
        )
        plt.tight_layout()
        roc_filename = os.path.join(output_dir, f"roc_curve_LR_{num_added}_synth_features.pdf")
        plt.savefig(roc_filename, bbox_inches='tight')
        print(f"Saved ROC curve to {roc_filename}")
        plt.close()
        roc_data_files.append(roc_data_path)

        # Calculate sensitivity at target specificity
        sensitivity_result_k = plotting.calculate_sensitivity_at_specificity(
            y,
            model_results_k['probabilities'],
            target_specificity=target_specificity
        )
        sensitivities.append({
            'num_synthetic': num_added,
            'n_features_total': len(current_features),
            'sensitivity': sensitivity_result_k['sensitivity'],
            'ci_low': sensitivity_result_k['ci_low'],
            'ci_high': sensitivity_result_k['ci_high'],
            'features': current_features.copy() # Store the list of features used
        })

    # Create DataFrame of sensitivity results
    sensitivity_df = pd.DataFrame(sensitivities)

    # Plot all ROC curves together and save to main plots directory
    fig_combined, ax_combined = plotting.plot_multiple_roc_curves(
        roc_data_files,
        labels=[f"LR with {n} synthetic features" for n in sensitivity_df['num_synthetic']],
        title=f"Comparison of ROC Curves with Added Synthetic Features (alpha={alpha}, ortho={orthogonality:.1f})",
        save_path=os.path.join(plots_dir, f"combined_roc_curves_alpha{alpha}_ortho{orthogonality:.1f}.pdf")
    )
    print(f"Saved combined ROC curves to {os.path.join(plots_dir, f'combined_roc_curves_alpha{alpha}_ortho{orthogonality:.1f}.pdf')}")
    plt.close()

    # Plot sensitivity vs number of features and save to main plots directory
    fig_sens, ax_sens = plt.subplots(figsize=(12, 7))

    # Sort sensitivity_df by num_synthetic for plotting
    sensitivity_df = sensitivity_df.sort_values('num_synthetic')

    # Plot sensitivity points with error bars
    ax_sens.errorbar(
        sensitivity_df['num_synthetic'],
        sensitivity_df['sensitivity'],
        yerr=[
            sensitivity_df['sensitivity'] - sensitivity_df['ci_low'],
            sensitivity_df['ci_high'] - sensitivity_df['sensitivity']
        ],
        fmt='o-',
        capsize=5,
        color='darkred',
        ecolor='gray',
        markersize=8,
        linewidth=2
    )

    # Add labels for each point
    for i, row in sensitivity_df.iterrows():
        ax_sens.annotate(
            f"{row['sensitivity']:.3f}",
            xy=(row['num_synthetic'], row['sensitivity']),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            fontsize=9
        )

    # Format the plot
    ax_sens.set_xlabel('Number of Synthetic Features Added')
    ax_sens.set_ylabel(f'Sensitivity at {target_specificity*100:.0f}% Specificity')
    ax_sens.set_title(f'Sensitivity vs Synthetic Features (alpha={alpha}, ortho={orthogonality:.1f})')
    ax_sens.grid(True, linestyle='--', alpha=0.7)
    # Ensure x-axis ticks cover the range of synthetic features added
    ax_sens.set_xticks(range(0, num_synthetic_features + 1, max(1, num_synthetic_features // 10)))

    # Save sensitivity plot to main plots directory
    sens_filename = os.path.join(plots_dir, f"sensitivity_vs_features_alpha{alpha}_ortho{orthogonality:.1f}.pdf")
    plt.savefig(sens_filename, bbox_inches='tight')
    print(f"Saved sensitivity plot to {sens_filename}")
    plt.close()

    # Save full results as pickle in the specific output directory
    results_path = os.path.join(output_dir, f'synthetic_results_alpha{alpha}_ortho{orthogonality:.1f}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'model_results': results,
            'sensitivity_results': sensitivity_df.to_dict('records'),
            'parameters': {
                'alpha': alpha,
                'orthogonality': orthogonality,
                'num_synthetic_features': num_synthetic_features
            }
        }, f)
    print(f"Saved full results to {results_path}")

    return {
        'model_results': results,
        'sensitivity_results': sensitivity_df
    }


# Global list of ID columns needed by models.py functions
id_columns = ['Patient_ID', 'Sample_ID', 'Tumor_type', 'AJCC_Stage']

def print_feature_correlations(X, y, feature_names):
    """
    Calculate and print the correlation between each feature and the target.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame containing features for the model.
    y : pandas.Series or numpy.ndarray
        Target variable (cancer vs normal).
    feature_names : list
        List of feature names to calculate correlations for.
    """
    print("\n==== Base Feature Correlations with Target ====")
    
    # Ensure y is a numpy array
    if isinstance(y, pd.Series):
        y_arr = y.values
    else:
        y_arr = y
        
    correlations = []
    
    # Calculate correlation for each feature
    for feature in feature_names:
        if feature in X.columns:
            # Calculate Pearson correlation
            feature_values = X[feature].values
            corr_matrix = np.corrcoef(feature_values, y_arr)
            correlation = corr_matrix[0, 1]
            
            # Store results
            correlations.append((feature, correlation))
    
    # Sort by absolute correlation value (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Print results in a table format
    print(f"{'Feature':<20} {'Correlation (alpha)':<15}")
    print("-" * 35)
    
    for feature, corr in correlations:
        print(f"{feature:<20} {corr:>15.4f}")
    
    print("\n")

def main():
    # Get command line arguments
    args = parse_arguments()

    # Map 'none' to None for log_transform
    log_transform = None if args.log_transform == 'none' else args.log_transform

    # Load the supplemental data
    print("Loading supplemental data...")
    supp_data_dict = supp_processor.load_all_supplemental_files()

    # Process and prepare data for model training
    print("\nProcessing clinical and protein data...")
    clinical_df = supp_data_dict['s4_clinical_characteristics']
    mutations_df = supp_data_dict['s5_mutations']
    protein_df = supp_data_dict['s6_protein_conc']

    # Create binary target: cancer vs. normal
    clinical_df['is_cancer'] = (clinical_df['Tumor_type'] != 'Normal').astype(int)

    # Fix Patient_ID format discrepancies
    clinical_df['Patient_ID'] = clinical_df['Patient_ID'].str.replace(r'[^\w\s_-]', '', regex=True)
    protein_df['Patient_ID'] = protein_df['Patient_ID'].str.replace(r'[^\w\s_-]', '', regex=True)
    mutations_df['Patient_ID'] = mutations_df['Patient_ID'].str.replace(r'[^\w\s_-]', '', regex=True)

    # Extract protein measurements
    # Ensure required protein columns from the paper exist
    original_protein_features = ['CA-125', 'CA19-9', 'CEA', 'HGF', 'Myeloperoxidase', 'OPN', 'Prolactin', 'TIMP-1']
    all_protein_columns = [col for col in protein_df.columns if col not in ['Patient_ID', 'Sample_ID', 'Tumor_type', 'AJCC_Stage', 'CancerSEEK_Logistic_Regression_Score', 'CancerSEEK_Test_Result']]
    protein_measurements = protein_df[['Patient_ID', 'Sample_ID'] + all_protein_columns].copy()

    # Merge protein data with clinical data
    # Use clinical_df as the base, ensuring all samples are included
    data_df = clinical_df.merge(protein_measurements, on=['Patient_ID', 'Sample_ID'], how='left')

    # Check for omega_score and add it
    omega_feature_name = None
    if 'omega_score' in mutations_df.columns:
        omega_feature_name = 'omega_score'
        omega_cols = ['Patient_ID', 'Sample_ID', omega_feature_name]
        # Ensure we handle potential duplicates if a patient/sample has multiple mutation entries
        omega_df = mutations_df[omega_cols].groupby(['Patient_ID', 'Sample_ID']).first().reset_index()
        # Merge with data_df
        data_df = data_df.merge(omega_df, on=['Patient_ID', 'Sample_ID'], how='left')
        # Fill missing omega_score with 0 (important for non-cancer or samples without mutation data)
        data_df[omega_feature_name] = data_df[omega_feature_name].fillna(0)
    else:
        print("Warning: omega_score not found in mutations data.")

    # Handle missing values for proteins (impute based on normal samples)
    normal_samples = data_df[data_df['Tumor_type'] == 'Normal']
    for col in all_protein_columns:
        if col in data_df.columns and data_df[col].isna().any():
            normal_mean = normal_samples[col].dropna().mean()
            # If all normal samples are NaN, use overall mean
            if pd.isna(normal_mean):
                normal_mean = data_df[col].dropna().mean()
            # If still NaN (e.g., all values are NaN), fill with 0 or handle appropriately
            if pd.isna(normal_mean):
                normal_mean = 0
            data_df[col] = data_df[col].fillna(normal_mean)
            # Check for any remaining NaNs after imputation
            if data_df[col].isna().any():
                 print(f"Warning: Column {col} still contains NaNs after imputation.")
                 data_df[col] = data_df[col].fillna(0) # Final fallback

    # Define base features for the model
    base_features = original_protein_features.copy()
    if omega_feature_name:
        base_features.append(omega_feature_name)

    # Verify all base features exist in the final dataframe
    missing_base = [f for f in base_features if f not in data_df.columns]
    if missing_base:
        raise ValueError(f"Error: The following base features are missing from the data: {missing_base}")

    # Create X (features + ID columns) and y (target)
    # Include ID columns needed by CV functions
    X = data_df[id_columns + base_features].copy()
    y = data_df['is_cancer']

    # Print correlations of base features with target
    print_feature_correlations(X, y, base_features)

    # Ensure Patient_ID and Sample_ID are set as index if needed by models.py, otherwise pass them in X
    # (Currently, models.py seems to expect them in X based on clinical_df handling)

    print(f"\nData prepared with {X.shape[0]} samples.")
    print(f"Base features ({len(base_features)}): {base_features}")
    print(f"Target alpha for synthetic features: {args.alpha}")
    print(f"Orthogonality parameter: {args.orthogonality}")
    print(f"Number of synthetic features to add: {args.num_synthetic_features}")

    # Train models by adding synthetic features
    results = train_models_with_synthetic_features(
        X_orig=data_df[id_columns + all_protein_columns + ([omega_feature_name] if omega_feature_name else [])].copy(), # Pass the full data_df for X_current_iter updates
        y=y,
        base_features=base_features,
        protein_features=original_protein_features, # Pass only the actual protein features here
        omega_feature=omega_feature_name,
        clinical_df=clinical_df,
        num_synthetic_features=args.num_synthetic_features,
        alpha=args.alpha,
        orthogonality=args.orthogonality,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        random_state=args.random_seed,
        standardize_features=args.standardize,
        log_transform=log_transform,
        target_specificity=args.target_specificity
    )

    # Display final sensitivity results
    print(f"\n==== Synthetic Feature Addition Analysis Complete ====")
    print(f"Target alpha: {args.alpha}")
    print(f"Orthogonality: {args.orthogonality}")
    print(f"\nSensitivity Results (Sensitivity at {args.target_specificity*100:.0f}% Specificity):")
    print(results['sensitivity_results'][['num_synthetic', 'n_features_total', 'sensitivity', 'ci_low', 'ci_high']])

    return results

# Example Usage (Optional - can be run if this script is executed directly)
# ... (Keep existing example usage for generate_synthetic_feature if desired)
# ... (Or add example usage for the main function)

if __name__ == '__main__':
    # Suppress convergence warnings
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice.")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")


    # Run the main experiment function
    main()