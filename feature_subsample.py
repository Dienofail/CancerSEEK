import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import sklearn
import os
import argparse
import pickle
from tqdm.auto import tqdm
import process_supplemental_data as supp_processor
import models
import plotting

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train logistic regression models with varying number of features')
    
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
    parser.add_argument('--model-type', type=str, choices=['LR', 'XGB', 'TF', 'MOE'], default='LR',
                        help='Model type for cancer detection (LR=Logistic Regression, XGB=XGBoost, TF=TensorFlow Fusion, MOE=Mixture of Experts) (default: LR)')

    # Set defaults
    parser.set_defaults(standardize=False)

    return parser.parse_args()

def train_models_with_feature_subsets(X, y, feature_importance_df, clinical_df, 
                                     outer_splits=10, inner_splits=4, 
                                     random_state=42, standardize_features=False, 
                                     log_transform=None, target_specificity=0.99,
                                     model_type='LR'):
    """
    Train models with varying numbers of features based on importance score.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame containing features for model
    y : pandas.Series or numpy.ndarray
        Target variable (cancer vs normal)
    feature_importance_df : pandas.DataFrame
        DataFrame with feature importance scores, sorted by importance
    clinical_df : pandas.DataFrame
        DataFrame with clinical information
    outer_splits : int, optional
        Number of folds for outer cross-validation (default: 10)
    inner_splits : int, optional
        Number of folds for inner cross-validation (default: 4)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    standardize_features : bool, optional
        Whether to standardize features (default: False)
    log_transform : str, optional
        Type of log transformation to apply to protein features ('log2', 'log10', or None)
    target_specificity : float, optional
        Target specificity for sensitivity calculations (default: 0.99)
    model_type : str, optional
        Model type to use ('LR', 'XGB', 'TF', or 'MOE') (default: 'LR')
        
    Returns:
    --------
    dict
        Dictionary containing results for models with different feature subsets
    """
    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}")
    
    # Create feature_subsample subfolder inside plots directory
    feature_subsample_dir = os.path.join(plots_dir, "feature_subsample")
    if not os.path.exists(feature_subsample_dir):
        os.makedirs(feature_subsample_dir)
        print(f"Created directory: {feature_subsample_dir}")
    
    # Sort features by importance score (descending)
    sorted_features = feature_importance_df.sort_values('Importance_Score', ascending=False)
    
    # Check if omega_score exists in X
    has_omega_score = 'omega_score' in X.columns
    
    # Initialize containers for results
    results = {}
    roc_data_files = []
    sensitivities = []
    
    # Min and max number of features to use
    min_features = 1
    max_features = len(sorted_features)
    
    # Train models with different feature subsets
    for n_features in tqdm(range(min_features, max_features + 1), desc=f"Training {model_type} models with feature subsets"):
        print(f"\n==== Training {model_type} model with {n_features} features ====")
        
        # Get top n features
        top_features = sorted_features['Feature'].iloc[:n_features].tolist()
        
        # Format features for protein_features
        # Some features may need normalization (Ω score -> omega_score)
        protein_features = []
        for feature in top_features:
            if feature == 'Ω score' and has_omega_score:
                # This will be handled separately
                continue
            elif feature.lower() == 'omega score' and has_omega_score:
                # This will be handled separately
                continue
            else:
                # Add to protein features list
                protein_features.append(feature)
        
        # Check if omega_score is in top features
        use_omega_score = 'Ω score' in top_features or 'omega score' in [f.lower() for f in top_features]
        
        # Debug info
        print(f"Using features: {top_features}")
        print(f"Protein features: {protein_features}")
        print(f"Using omega_score: {use_omega_score}")
        
        # Train model with nested cross-validation based on model type
        if model_type == 'TF':
            # Use TensorFlow fusion model
            model_results = models.nested_cross_validation_tf(
                X, y, 
                protein_features=protein_features,
                omega_score_col='omega_score' if use_omega_score and has_omega_score else None,
                clinical_df=clinical_df,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                random_state=random_state,
                standardize_features=standardize_features,
                log_transform=log_transform,
                is_multiclass=False
            )
        elif model_type == 'MOE':
            # Use Mixture of Experts model
            model_results = models.nested_cross_validation_moe(
                X, y, 
                protein_features=protein_features,
                omega_score_col='omega_score' if use_omega_score and has_omega_score else None,
                clinical_df=clinical_df,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                random_state=random_state,
                standardize_features=standardize_features,
                log_transform=log_transform,
                is_multiclass=False
            )
        elif model_type == 'LR':
            model_results = models.nested_cross_validation(
                X, y, 
                protein_features=protein_features,
                omega_score_col='omega_score' if use_omega_score and has_omega_score else None,
                clinical_df=clinical_df,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                random_state=random_state,
                standardize_features=standardize_features,
                log_transform=log_transform,
                model_type=model_type,
                max_iter=1000  # Increase max_iter to help with convergence warnings
            )
        else:  # XGB
            model_results = models.nested_cross_validation(
                X, y, 
                protein_features=protein_features,
                omega_score_col='omega_score' if use_omega_score and has_omega_score else None,
                clinical_df=clinical_df,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                random_state=random_state,
                standardize_features=standardize_features,
                log_transform=log_transform,
                model_type=model_type
            )
        
        # Store results
        results[n_features] = model_results
        
        # Save individual ROC curve and data to feature_subsample directory
        roc_data_path = os.path.join(feature_subsample_dir, f"roc_curve_data_{model_type}_{n_features}_features.pkl")
        
        # Plot and save ROC curve
        fig_roc, ax_roc = plotting.plot_roc_curve(
            y, 
            model_results['probabilities'],
            title=f"ROC Curve for CancerSEEK {model_type} with {n_features} features",
            model_type=f"{model_type}-{n_features}",
            save_data=roc_data_path
        )
        plt.tight_layout()
        # Save ROC curve to PDF in feature_subsample directory
        roc_filename = os.path.join(feature_subsample_dir, f"roc_curve_{model_type}_{n_features}_features.pdf")
        plt.savefig(roc_filename, bbox_inches='tight')
        print(f"Saved ROC curve to {roc_filename}")
        plt.close()
        
        # Add to list of ROC data files (use the path to feature_subsample directory)
        roc_data_files.append(roc_data_path)
        
        # Calculate sensitivity at target specificity
        sensitivity_result = plotting.calculate_sensitivity_at_specificity(
            y,
            model_results['probabilities'],
            target_specificity=target_specificity
        )
        
        # Store sensitivity results with the number of features
        sensitivity_info = {
            'n_features': n_features,
            'sensitivity': sensitivity_result['sensitivity'],
            'ci_low': sensitivity_result['ci_low'],
            'ci_high': sensitivity_result['ci_high'],
            'features': top_features
        }
        sensitivities.append(sensitivity_info)
    
    # Create DataFrame of sensitivity results
    sensitivity_df = pd.DataFrame(sensitivities)
    
    # Plot all ROC curves together and save to main plots directory
    fig_combined, ax_combined = plotting.plot_multiple_roc_curves(
        roc_data_files,
        labels=[f"{model_type} with {n} features" for n in range(min_features, max_features + 1)],
        title=f"Comparison of ROC Curves with Different Number of Features ({model_type})",
        save_path=os.path.join(plots_dir, f"combined_roc_curves_{model_type}.pdf")
    )
    print(f"Saved combined ROC curves to {os.path.join(plots_dir, f'combined_roc_curves_{model_type}.pdf')}")
    plt.close()
    
    # Plot sensitivity vs number of features and save to main plots directory
    fig_sens, ax_sens = plt.subplots(figsize=(10, 6))
    
    # Plot sensitivity points with error bars
    ax_sens.errorbar(
        sensitivity_df['n_features'],
        sensitivity_df['sensitivity'],
        yerr=[
            sensitivity_df['sensitivity'] - sensitivity_df['ci_low'],
            sensitivity_df['ci_high'] - sensitivity_df['sensitivity']
        ],
        fmt='o-',
        capsize=5,
        color='steelblue',
        ecolor='gray',
        markersize=8,
        linewidth=2
    )
    
    # Add labels for each point
    for i, row in sensitivity_df.iterrows():
        ax_sens.annotate(
            f"{row['sensitivity']:.3f}",
            xy=(row['n_features'], row['sensitivity']),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            fontsize=9
        )
    
    # Format the plot
    ax_sens.set_xlabel('Number of Features')
    ax_sens.set_ylabel(f'Sensitivity at {target_specificity*100:.0f}% Specificity')
    ax_sens.set_title(f'Sensitivity at {target_specificity*100:.0f}% Specificity vs Number of Features ({model_type})')
    ax_sens.grid(True, linestyle='--', alpha=0.7)
    ax_sens.set_xticks(range(min_features, max_features + 1))
    
    # Save sensitivity plot to main plots directory
    sens_filename = os.path.join(plots_dir, f"sensitivity_vs_features_{int(target_specificity*100)}_specificity_{model_type}.pdf")
    plt.savefig(sens_filename, bbox_inches='tight')
    print(f"Saved sensitivity plot to {sens_filename}")
    plt.close()
    
    # Save full results as pickle in feature_subsample directory
    with open(os.path.join(feature_subsample_dir, f'feature_subsample_results_{model_type}.pkl'), 'wb') as f:
        pickle.dump({
            'model_results': results,
            'sensitivity_results': sensitivity_df.to_dict('records')
        }, f)
    
    return {
        'model_results': results,
        'sensitivity_results': sensitivity_df
    }

def main():
    # Get command line arguments
    args = parse_arguments()
    
    # Map 'none' to None for log_transform
    log_transform = None if args.log_transform == 'none' else args.log_transform
    
    # Load the supplemental data
    print("Loading supplemental data...")
    supp_data_dict = supp_processor.load_all_supplemental_files()
    
    # Access the model coefficients data from S9
    lg_model_coef = supp_data_dict['s9_logistic_regression']
    print("\nModel coefficients data:")
    print(lg_model_coef.head())
    
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
    protein_columns = [col for col in protein_df.columns if col not in ['Patient_ID', 'Sample_ID', 'Tumor_type', 'AJCC_Stage', 'CancerSEEK_Logistic_Regression_Score', 'CancerSEEK_Test_Result']]
    protein_measurements = protein_df[['Patient_ID', 'Sample_ID'] + protein_columns].copy()
    
    # Merge protein data with clinical data
    data_df = clinical_df.merge(protein_measurements, on=['Patient_ID', 'Sample_ID'], how='left')
    
    # Check for omega_score and add it
    if 'omega_score' in mutations_df.columns:
        omega_cols = ['Patient_ID', 'Sample_ID', 'omega_score']
        omega_df = mutations_df[omega_cols].drop_duplicates()
        # Merge with data_df
        data_df = data_df.merge(omega_df, on=['Patient_ID', 'Sample_ID'], how='left')
        # Fill missing omega_score with 0
        data_df['omega_score'] = data_df['omega_score'].fillna(0)
    
    # Handle missing values
    # Filter to get only normal samples
    normal_samples = data_df[data_df['Tumor_type'] == 'Normal']
    
    # Impute protein values using mean of normal samples
    for col in protein_columns:
        if col in data_df.columns and data_df[col].isna().any():
            normal_mean = normal_samples[col].dropna().mean()
            if pd.isna(normal_mean):
                normal_mean = data_df[col].dropna().mean()
            data_df[col] = data_df[col].fillna(normal_mean)
    
    # Impute omega_score if needed
    if 'omega_score' in data_df.columns and data_df['omega_score'].isna().any():
        normal_omega_mean = normal_samples['omega_score'].dropna().mean()
        if pd.isna(normal_omega_mean):
            normal_omega_mean = data_df['omega_score'].dropna().mean()
        data_df['omega_score'] = data_df['omega_score'].fillna(normal_omega_mean)
    
    # Create list of columns to keep for modeling
    id_columns = ['Patient_ID', 'Sample_ID', 'Tumor_type', 'AJCC_Stage']
    
    # Identify feature columns
    feature_columns = []
    for col in data_df.columns:
        if (col not in id_columns and 
            col not in ['Primary_tumor_sample_ID', 'Histopathology', 'Race', 'Sex', 
                       'Plasma_volume_(mL)', 'Plasma_DNA_concentration', 
                       'CancerSEEK_Logistic_Regression_Score', 'CancerSEEK_Test_Result', 
                       'is_cancer'] and
            pd.api.types.is_numeric_dtype(data_df[col])):
            feature_columns.append(col)
    
    # Create X with both ID columns and feature columns
    X = data_df[id_columns + feature_columns].copy()
    y = data_df['is_cancer']
    
    print(f"\nData prepared with {X.shape[0]} samples and {len(feature_columns)} feature columns")
    
    # Train models with different feature subsets
    results = train_models_with_feature_subsets(
        X=X, 
        y=y, 
        feature_importance_df=lg_model_coef,
        clinical_df=clinical_df,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        random_state=args.random_seed,
        standardize_features=args.standardize,
        log_transform=log_transform,
        target_specificity=args.target_specificity,
        model_type=args.model_type
    )
    
    # Display final results
    print(f"\n==== Feature Subset Analysis Complete for {args.model_type} ====")
    print("\nSensitivity Results:")
    print(results['sensitivity_results'])
    
    return results

if __name__ == "__main__":
    # Suppress convergence warnings
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    main()
