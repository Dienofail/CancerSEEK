import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # Import tqdm for progress bars that work in both CLI and Jupyter
# Try to import TensorFlow, but make it optional
# TensorFlow is only needed if using the 'TF' model type
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers, models, callbacks, regularizers, optimizers, utils
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not found. TF model type will not be available.")
    print("To use TensorFlow models, install with: pip install tensorflow")

def normalize_protein_data(protein_df, clinical_df, detection_limits=None, training_indices=None, percentile_values=None):
    """
    Normalize protein data according to the specified method:
    1. Set values below the maximum lower detection limit to that limit
    2. Set values above the minimum upper detection limit to that limit
    3. Set values below the 95th percentile of normal samples to zero
    
    Parameters:
    -----------
    protein_df : pandas.DataFrame
        DataFrame containing protein measurements
    clinical_df : pandas.DataFrame
        DataFrame containing clinical information with Tumor_type column
    detection_limits : dict, optional
        Dictionary with protein names as keys and tuples of (lower_limit, upper_limit) as values
        If None, detection limits will be inferred from data
    training_indices : list, optional
        Indices to use as training set for calculating the 95th percentile
        If None, all normal samples will be used
    percentile_values : dict, optional
        Pre-calculated 95th percentile values for each protein from training data
        If provided, these values will be used instead of calculating from training data
        
    Returns:
    --------
    tuple
        (normalized_df, percentile_values_dict)
        normalized_df: pandas.DataFrame - Normalized protein data
        percentile_values_dict: dict - 95th percentile values for each protein
    """
    # Create a copy of the input data to avoid modifying the original
    normalized_df = protein_df.copy()
    
    # Identify protein columns (exclude non-protein columns)
    non_protein_cols = ['Patient_ID', 'Sample_ID', 'Tumor_type', 'AJCC_Stage', 
                        'CancerSEEK_Logistic_Regression_Score', 'CancerSEEK_Test_Result']
    protein_cols = [col for col in normalized_df.columns if col not in non_protein_cols]
    
    # Identify normal samples from clinical data
    normal_samples = clinical_df[clinical_df['Tumor_type'] == 'Normal']['Patient_ID'].tolist()
    
    # Check if Patient_ID exists in the dataframe, otherwise use index
    if 'Patient_ID' in normalized_df.columns:
        normal_indices = normalized_df[normalized_df['Patient_ID'].isin(normal_samples)].index
    else:
        print("Warning: 'Patient_ID' column not found in protein dataframe. Using training_indices directly.")
        normal_indices = training_indices if training_indices is not None else []
    
    if training_indices is None:
        training_indices = normal_indices
    
    # If detection limits are not provided, create a placeholder
    if detection_limits is None:
        detection_limits = {}
        for protein in protein_cols:
            # This is a placeholder. In practice, you would need actual detection limits
            # Here we use min and max as proxies for lower and upper detection limits
            lower_limit = normalized_df[protein].min()
            upper_limit = normalized_df[protein].max()
            detection_limits[protein] = (lower_limit, upper_limit)
    
    # Initialize percentile_values_dict if not provided
    if percentile_values is None:
        percentile_values_dict = {}
    else:
        percentile_values_dict = percentile_values.copy()
    
    # Process each protein
    for protein in protein_cols:
        # Get the detection limits for this protein
        lower_limit, upper_limit = detection_limits.get(protein, (normalized_df[protein].min(), normalized_df[protein].max()))
        
        # Apply the first part of normalization (clipping values to detection limits)
        normalized_df[protein] = normalized_df[protein].clip(lower=lower_limit, upper=upper_limit)
        
        # Calculate or use provided 95th percentile
        if protein not in percentile_values_dict:
            # Calculate the 95th percentile among normal samples in the training set
            # Check if we have valid training indices before calculating
            if len(training_indices) > 0:
                percentile_values_dict[protein] = normalized_df.loc[training_indices, protein].quantile(0.95)
            else:
                # Fallback to overall 95th percentile if no training indices
                percentile_values_dict[protein] = normalized_df[protein].quantile(0.95)
        
        # Apply the second part of normalization (set to 0 if below 95th percentile of normal samples)
        normalized_df[protein] = normalized_df[protein].apply(
            lambda x: 0 if x < percentile_values_dict[protein] else x
        )
    
    return normalized_df, percentile_values_dict

def normalize_mutation_data(mutation_df, clinical_df, detection_limits=None, training_indices=None, stored_limits=None):
    """
    Normalize mutation data according to the specified method:
    1. Set values below the maximum lower detection limit to that limit
    2. Set values above the minimum upper detection limit to that limit
    
    Parameters:
    -----------
    mutation_df : pandas.DataFrame
        DataFrame containing mutation measurements with lambda_score column
    clinical_df : pandas.DataFrame
        DataFrame containing clinical information with Tumor_type column
    detection_limits : tuple, optional
        Tuple of (lower_limit, upper_limit) for lambda_score
        If None, detection limits will be inferred from data
    training_indices : list, optional
        Indices to use as training set
        If None, all normal samples will be used
    stored_limits : tuple, optional
        Pre-calculated detection limits from training data
        If provided, these limits will be used instead of calculating from data
        
    Returns:
    --------
    tuple
        (normalized_df, detection_limits)
        normalized_df: pandas.DataFrame - Normalized mutation data
        detection_limits: tuple - (lower_limit, upper_limit) for lambda_score
    """
    # Create a copy of the input data to avoid modifying the original
    normalized_df = mutation_df.copy()
    
    # Identify normal samples from clinical data
    normal_samples = clinical_df[clinical_df['Tumor_type'] == 'Normal']['Patient_ID'].tolist()
    
    # Check if Patient_ID exists in the dataframe, otherwise use index
    if 'Patient_ID' in normalized_df.columns:
        normal_indices = normalized_df[normalized_df['Patient_ID'].isin(normal_samples)].index
    else:
        print("Warning: 'Patient_ID' column not found in mutation dataframe. Using training_indices directly.")
        normal_indices = training_indices if training_indices is not None else []
    
    if training_indices is None:
        training_indices = normal_indices
    
    # If stored limits are provided, use them
    if stored_limits is not None:
        lower_limit, upper_limit = stored_limits
    # If detection limits are not provided, create from data
    elif detection_limits is None:
        # Use min and max as proxies for lower and upper detection limits
        # But only calculate from training indices to prevent data leakage
        if len(training_indices) > 0 and 'omega_score' in normalized_df.columns:
            training_data = normalized_df.loc[training_indices, 'omega_score']
            lower_limit = training_data.min()
            upper_limit = training_data.max()
        else:
            # Fallback if no training indices or omega_score column
            if 'omega_score' in normalized_df.columns:
                lower_limit = normalized_df['omega_score'].min()
                upper_limit = normalized_df['omega_score'].max()
            else:
                print("Warning: 'omega_score' column not found. Using defaults.")
                lower_limit, upper_limit = 0, 1
        detection_limits = (lower_limit, upper_limit)
    else:
        lower_limit, upper_limit = detection_limits
    
    # Apply normalization (clipping values to detection limits)
    if 'omega_score' in normalized_df.columns:
        normalized_df['omega_score'] = normalized_df['omega_score'].clip(lower=lower_limit, upper=upper_limit)
    else:
        print("Warning: 'omega_score' column not found. Skipping normalization.")
    
    return normalized_df, (lower_limit, upper_limit)

def filter_proteins_by_mwu_test(protein_df, clinical_df):
    """
    Filter proteins using Mann-Whitney U test to keep only those with
    higher median values in cancer samples compared to normal samples.
    
    Parameters:
    -----------
    protein_df : pandas.DataFrame
        DataFrame containing protein measurements
    clinical_df : pandas.DataFrame
        DataFrame containing clinical information with Tumor_type column
        
    Returns:
    --------
    list
        List of protein names that pass the filter
    """
    # Identify protein columns (exclude non-protein columns)
    non_protein_cols = ['Patient_ID', 'Sample_ID', 'Tumor_type', 'AJCC_Stage', 
                        'CancerSEEK_Logistic_Regression_Score', 'CancerSEEK_Test_Result']
    protein_cols = [col for col in protein_df.columns if col not in non_protein_cols]
    
    # Identify normal vs cancer samples
    normal_samples = clinical_df[clinical_df['Tumor_type'] == 'Normal']['Patient_ID'].tolist()
    cancer_samples = clinical_df[clinical_df['Tumor_type'] != 'Normal']['Patient_ID'].tolist()
    
    # Create masks for normal and cancer samples
    normal_mask = protein_df['Patient_ID'].isin(normal_samples)
    cancer_mask = protein_df['Patient_ID'].isin(cancer_samples)
    
    # Store proteins that pass the filter
    passed_proteins = []
    
    # Perform Mann-Whitney U test for each protein
    for protein in protein_cols:
        # Get protein values for normal and cancer samples
        normal_values = protein_df.loc[normal_mask, protein].dropna()
        cancer_values = protein_df.loc[cancer_mask, protein].dropna()
        
        # Calculate medians
        normal_median = normal_values.median()
        cancer_median = cancer_values.median()
        
        # Only keep proteins with higher median in cancer samples
        if cancer_median > normal_median:
            # Perform Mann-Whitney U test
            stat, p_value = mannwhitneyu(cancer_values, normal_values, alternative='greater')
            passed_proteins.append({
                'protein': protein,
                'normal_median': normal_median,
                'cancer_median': cancer_median,
                'p_value': p_value,
                'statistic': stat
            })
    
    # Convert to a DataFrame for easier inspection and sort by p-value
    result_df = pd.DataFrame(passed_proteins).sort_values('p_value')
    
    # Extract just the protein names
    filtered_proteins = result_df['protein'].tolist()
    
    return filtered_proteins, result_df

def stratified_fold_assignment(clinical_df, n_splits=10):
    """
    Assign fold IDs to patients in a way that preserves distribution of
    tumor types, age groups, gender, and race across folds.
    
    Parameters:
    -----------
    clinical_df : pandas.DataFrame
        DataFrame containing clinical information with demographic details
    n_splits : int, optional
        Number of folds to create (default: 10)
        
    Returns:
    --------
    pandas.Series
        Series with patient IDs as index and fold assignments as values
    """
    # Create a copy to avoid modifying the original
    df = clinical_df.copy()
    
    # Create age groups
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 40, 50, 60, 70, 100], 
                             labels=['<40', '40-50', '50-60', '60-70', '>70'])
    
    # Create a stratification variable combining tumor type, age group, sex, and race
    df['Stratify'] = df['Tumor_type'] + '_' + df['Age_Group'].astype(str) + '_' + df['Sex'] + '_' + df['Race']
    
    # Initialize fold assignments
    fold_assignments = pd.Series(index=df['Patient_ID'], dtype=int)
    
    # Get the distribution of stratification variables
    strata_counts = df['Stratify'].value_counts()
    
    # Sort strata by frequency (rarest first) to ensure they're evenly distributed
    sorted_strata = strata_counts.sort_values().index
    
    # Current count of samples in each fold
    fold_counts = np.zeros(n_splits)
    
    # Process each stratum, starting with the rarest
    for stratum in sorted_strata:
        # Get patients in this stratum
        stratum_patients = df[df['Stratify'] == stratum]['Patient_ID'].values
        
        # Assign patients to folds, trying to keep fold sizes balanced
        for patient in stratum_patients:
            # Find the fold with the fewest samples
            min_fold = np.argmin(fold_counts)
            fold_assignments[patient] = min_fold
            fold_counts[min_fold] += 1
    
    return fold_assignments

def perform_cross_validation(X, y, protein_features, omega_score_col='omega_score', clinical_df=None, n_splits=10, random_state=42,
                      standardize_features=True, log_transform=None):
    """
    Perform 10-fold cross-validation of logistic regression model.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame containing features for model
    y : pandas.Series or numpy.ndarray
        Target variable (cancer vs normal)
    protein_features : list
        List of protein features to use in the model
    omega_score_col : str, optional
        Name of the omega score column (default: 'omega_score')
    clinical_df : pandas.DataFrame, optional
        DataFrame with clinical information for stratified sampling
    n_splits : int, optional
        Number of folds for cross-validation (default: 10)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    standardize_features : bool, optional
        Whether to standardize features (default: True)
    log_transform : str, optional
        Type of log transformation to apply to protein features ('log2', 'log10', or None)
        
    Returns:
    --------
    dict
        Dictionary containing cross-validation results
    """
    # Include omega score with protein features
    features = protein_features.copy()
    if omega_score_col in X.columns:
        features.append(omega_score_col)
    
    # Create feature matrix using only selected features
    X_selected = X[features].copy()
    
    # Apply log transformation to protein features if specified
    if log_transform in ['log2', 'log10']:
        for feature in protein_features:
            if feature in X_selected.columns:
                # Add a small constant to avoid log(0)
                if log_transform == 'log2':
                    X_selected[feature] = np.log2(X_selected[feature] + 1e-10)
                elif log_transform == 'log10':
                    X_selected[feature] = np.log10(X_selected[feature] + 1e-10)
    
    # Standardize features if requested
    if standardize_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)
    else:
        X_scaled = X_selected.values
        X_scaled_df = X_selected
    
    # Initialize result containers
    fold_results = []
    all_predictions = np.zeros_like(y, dtype=float)
    all_probabilities = np.zeros_like(y, dtype=float)
    feature_importances = pd.DataFrame(0, index=features, columns=['importance'])
    
    # Create either stratified folds or use pre-assigned folds
    if clinical_df is not None:
        # Use patient-level stratification if clinical data is provided
        fold_assignments = stratified_fold_assignment(clinical_df, n_splits=n_splits)
        unique_indices = X.index.unique()
        fold_indices = []
        
        # Convert fold assignments to indices for cross-validation
        for fold_id in range(n_splits):
            fold_patients = fold_assignments[fold_assignments == fold_id].index
            fold_indices.append([i for i, idx in enumerate(X.index) 
                                if idx in fold_patients])
        
        kf = KFold(n_splits=n_splits, shuffle=False)
        kf.split = lambda X, y=None, groups=None: iter(fold_indices)
    else:
        # Use regular stratified K-fold if no clinical data is provided
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled, y)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model with all features
        model = LogisticRegression(penalty='elasticnet', solver='saga', 
                                   l1_ratio=0.5, C=1.0, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Store predictions and probabilities
        all_predictions[test_idx] = y_pred
        all_probabilities[test_idx] = y_prob
        
        # Calculate performance metrics
        fold_acc = accuracy_score(y_test, y_pred)
        fold_auc = roc_auc_score(y_test, y_prob)
        
        # Store fold results
        fold_results.append({
            'fold': fold,
            'accuracy': fold_acc,
            'auc': fold_auc,
            'test_indices': test_idx
        })
        
        # Assess feature importance by dropping one feature at a time
        for i, feature in enumerate(features):
            # Create a copy of the training data without one feature
            X_train_dropped = np.delete(X_train, i, axis=1)
            X_test_dropped = np.delete(X_test, i, axis=1)
            
            # Train model without this feature
            model_dropped = LogisticRegression(penalty='elasticnet', solver='saga', 
                                              l1_ratio=0.5, C=1.0, random_state=random_state)
            model_dropped.fit(X_train_dropped, y_train)
            
            # Calculate accuracy
            y_pred_dropped = model_dropped.predict(X_test_dropped)
            acc_dropped = accuracy_score(y_test, y_pred_dropped)
            
            # Increase importance if dropping the feature decreases accuracy
            feature_importances.loc[feature, 'importance'] += (fold_acc - acc_dropped)
    
    # Calculate overall metrics
    overall_acc = accuracy_score(y, all_predictions)
    overall_auc = roc_auc_score(y, all_probabilities)
    
    # Average feature importances across folds
    feature_importances['importance'] = feature_importances['importance'] / n_splits
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    
    return {
        'fold_results': fold_results,
        'overall_accuracy': overall_acc,
        'overall_auc': overall_auc,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'feature_importances': feature_importances
    }

def nested_cross_validation(X, y, protein_features, omega_score_col='omega_score', 
                            clinical_df=None, outer_splits=10, inner_splits=5, random_state=42,
                            standardize_features=True, log_transform=None, model_type='LR',
                            custom_model=None):
    """
    Perform nested cross-validation for hyperparameter optimization.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame containing features for model
    y : pandas.Series or numpy.ndarray
        Target variable (cancer vs normal)
    protein_features : list
        List of protein features to use in the model
    omega_score_col : str, optional
        Name of the omega score column (default: 'omega_score')
    clinical_df : pandas.DataFrame, optional
        DataFrame with clinical information for stratified sampling
    outer_splits : int, optional
        Number of folds for outer cross-validation (default: 10)
    inner_splits : int, optional
        Number of folds for inner cross-validation (default: 5)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    standardize_features : bool, optional
        Whether to standardize features (default: True)
    log_transform : str, optional
        Type of log transformation to apply to protein features ('log2', 'log10', or None)
    model_type : str, optional
        Type of model to use ('LR' for Logistic Regression, 'XGB' for XGBoost, default: 'LR')
    custom_model : object, optional
        Custom model instance to use instead of creating a new one (default: None)
        
    Returns:
    --------
    dict
        Dictionary containing nested cross-validation results
    """
    print(f"Starting nested cross-validation for cancer detection using {model_type}...")
    # Include omega score with protein features
    features = protein_features.copy()
    if omega_score_col in X.columns:
        features.append(omega_score_col)
    
    # Parameter grid for hyperparameter optimization based on model type
    if model_type == 'LR':
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'l1_ratio': [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        }
        # Use custom model if provided, otherwise create a new one
        if custom_model is not None:
            base_model = custom_model
        else:
            base_model = LogisticRegression(penalty='elasticnet', solver='saga', random_state=random_state)
    elif model_type == 'XGB':
        param_grid = {
            'n_estimators': [10, 25, 50, 100, 200],
            'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
            'max_depth': [5, 7, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        # Use custom model if provided, otherwise create a new one
        if custom_model is not None:
            base_model = custom_model
        else:
            base_model = XGBClassifier(random_state=random_state, eval_metric='mlogloss')
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'LR' or 'XGB'.")
    
    # Create outer folds
    if clinical_df is not None:
        print("Creating stratified folds based on clinical data...")
        fold_assignments = stratified_fold_assignment(clinical_df, n_splits=outer_splits)
        unique_indices = X.index.unique()
        fold_indices = []
        
        for fold_id in range(outer_splits):
            fold_patients = fold_assignments[fold_assignments == fold_id].index
            fold_idx = [i for i, idx in enumerate(X.index) 
                            if idx in fold_patients]
            # Only add non-empty fold indices
            if len(fold_idx) > 0:
                fold_indices.append(fold_idx)
        
        # Make sure we have at least one valid train/test split
        if len(fold_indices) >= 2:
            # Create train/test splits from the fold indices
            train_test_splits = []
            
            for i in range(len(fold_indices)):
                test_idx = fold_indices[i]
                train_idx = [idx for j, fold in enumerate(fold_indices) for idx in fold if j != i]
                train_test_splits.append((train_idx, test_idx))
            
            outer_cv = KFold(n_splits=len(train_test_splits), shuffle=False)
            outer_cv.split = lambda X, y=None, groups=None: iter(train_test_splits)
        else:
            # Not enough valid folds, fall back to StratifiedKFold
            print("Warning: Not enough valid folds from clinical data, falling back to StratifiedKFold.")
            outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    else:
        outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    
    # Initialize result containers
    fold_results = []
    all_predictions = np.zeros_like(y, dtype=float)
    all_probabilities = np.zeros_like(y, dtype=float)
    
    # For ROC curve plotting
    fpr_list = []
    tpr_list = []
    
    # Prepare splits for tqdm
    splits = list(outer_cv.split(X.values, y))
    
    # Separate protein data and mutation data
    protein_cols = protein_features
    X_protein = X[protein_cols]
    
    # Check if mutation data (omega score) exists
    has_mutation_data = omega_score_col in X.columns
    if has_mutation_data:
        X_mutation = X[[omega_score_col]]
    
    # Perform nested cross-validation
    for fold, (train_idx, test_idx) in enumerate(tqdm(splits, desc=f"{model_type} cancer detection CV folds")):
        # Access the original data for proper normalization
        X_train_orig = X.iloc[train_idx].copy()
        X_test_orig = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Normalize protein data - prevents data leakage by using only training data for normalization parameters
        X_protein_train = X_protein.iloc[train_idx].copy()
        X_protein_test = X_protein.iloc[test_idx].copy()
        
        # Use numerical indices for normalization to avoid KeyError
        # Create index mapping for train set
        train_indices = list(range(len(X_protein_train)))
        
        # Normalize protein data using numerical indices
        X_protein_train_norm, percentile_values = normalize_protein_data(
            X_protein_train.reset_index(drop=True), clinical_df, training_indices=train_indices)
        
        # Apply same normalization to test data using parameters from training
        X_protein_test_norm, _ = normalize_protein_data(
            X_protein_test.reset_index(drop=True), clinical_df, training_indices=None, percentile_values=percentile_values)
        
        # Normalize mutation data if available
        if has_mutation_data:
            X_mutation_train = X_mutation.iloc[train_idx].copy()
            X_mutation_test = X_mutation.iloc[test_idx].copy()
            
            # Normalize mutation data using numerical indices
            X_mutation_train_norm, detection_limits = normalize_mutation_data(
                X_mutation_train.reset_index(drop=True), clinical_df, training_indices=train_indices)
            
            # Apply same normalization to test data using parameters from training
            X_mutation_test_norm, _ = normalize_mutation_data(
                X_mutation_test.reset_index(drop=True), clinical_df, training_indices=None, stored_limits=detection_limits)
            
            # Combine protein and mutation features
            X_train_combined = pd.concat([X_protein_train_norm, X_mutation_train_norm], axis=1)
            X_test_combined = pd.concat([X_protein_test_norm, X_mutation_test_norm], axis=1)
        else:
            # Only use protein features
            X_train_combined = X_protein_train_norm
            X_test_combined = X_protein_test_norm
        
        # Use only selected features
        X_train_selected = X_train_combined[features].copy()
        X_test_selected = X_test_combined[features].copy()
        
        # Apply log transformation to protein features if specified
        if log_transform in ['log2', 'log10']:
            for feature in protein_features:
                if feature in X_train_selected.columns:
                    # Add a small constant to avoid log(0)
                    if log_transform == 'log2':
                        X_train_selected[feature] = np.log2(X_train_selected[feature] + 1e-10)
                        X_test_selected[feature] = np.log2(X_test_selected[feature] + 1e-10)
                    elif log_transform == 'log10':
                        X_train_selected[feature] = np.log10(X_train_selected[feature] + 1e-10)
                        X_test_selected[feature] = np.log10(X_test_selected[feature] + 1e-10)
        
        # Standardize features if requested
        if standardize_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
        else:
            X_train_scaled = X_train_selected.values
            X_test_scaled = X_test_selected.values
        
        # Inner cross-validation for hyperparameter optimization
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        
        # Setup grid search with inner CV
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        # Find best hyperparameters
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions on test set
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Store predictions and probabilities
        all_predictions[test_idx] = y_pred
        all_probabilities[test_idx] = y_prob
        
        # Calculate performance metrics
        fold_acc = accuracy_score(y_test, y_pred)
        fold_auc = roc_auc_score(y_test, y_prob)
        
        # Calculate ROC curve data for this fold
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        
        # Store fold results
        fold_results.append({
            'fold': fold,
            'accuracy': fold_acc,
            'auc': fold_auc,
            'best_params': grid_search.best_params_,
            'test_indices': test_idx
        })
    
    # Calculate overall metrics
    overall_acc = accuracy_score(y, all_predictions)
    overall_auc = roc_auc_score(y, all_probabilities)
    
    # Calculate mean ROC curve for plotting
    from sklearn.metrics import roc_curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    
    for i in range(len(fpr_list)):
        mean_tpr += np.interp(mean_fpr, fpr_list[i], tpr_list[i])
    
    mean_tpr /= len(fpr_list)
    
    print(f"Completed cancer detection model with overall AUC: {overall_auc:.4f}")
    
    return {
        'fold_results': fold_results,
        'overall_accuracy': overall_acc,
        'overall_auc': overall_auc,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'roc_data': {
            'mean_fpr': mean_fpr,
            'mean_tpr': mean_tpr,
            'fpr_list': fpr_list,
            'tpr_list': tpr_list
        }
    }

def nested_cross_validation_rf(X, y, protein_features=None, omega_score_col='omega_score', 
                               include_gender=True, clinical_df=None, outer_splits=10, 
                               inner_splits=6, random_state=42, standardize_features=True,
                               log_transform=None, model_type='RF'):
    """
    Perform nested cross-validation for random forest or XGBoost tissue localization model.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame containing features for model
    y : pandas.Series or numpy.ndarray
        Target variable (cancer types)
    protein_features : list, optional
        List of protein features to use (default: None, use all available)
    omega_score_col : str, optional
        Name of the omega score column (default: 'omega_score')
    include_gender : bool, optional
        Whether to include patient gender (default: True)
    clinical_df : pandas.DataFrame, optional
        DataFrame with clinical information for stratified sampling
    outer_splits : int, optional
        Number of folds for outer cross-validation (default: 10)
    inner_splits : int, optional
        Number of folds for inner cross-validation (default: 5)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    standardize_features : bool, optional
        Whether to standardize features (default: True)
    log_transform : str, optional
        Type of log transformation to apply to protein features ('log2', 'log10', or None)
    model_type : str, optional
        Type of model to use ('RF' for Random Forest, 'XGB' for XGBoost, default: 'RF')
        
    Returns:
    --------
    dict
        Dictionary containing nested cross-validation results
    """
    print(f"Starting nested cross-validation for tissue localization using {model_type}...")
    # If protein features not specified, use all available
    if protein_features is None:
        non_protein_cols = ['Patient_ID', 'Sample_ID', 'Tumor_type', 'AJCC_Stage', 
                            'CancerSEEK_Logistic_Regression_Score', 'CancerSEEK_Test_Result',
                            'is_cancer', 'Age', 'Race', 'Sex']
        protein_features = [col for col in X.columns if col not in non_protein_cols 
                         and col != omega_score_col]
    
    # Create list of features to use
    features = protein_features.copy()
    if omega_score_col in X.columns:
        features.append(omega_score_col)
    
    # Add gender if requested and available
    gender_col = 'Sex'
    has_gender = gender_col in X.columns
    
    # Separate protein data and mutation data
    X_protein = X[protein_features].copy()
    
    # Check if mutation data (omega score) exists
    has_mutation_data = omega_score_col in X.columns
    if has_mutation_data:
        X_mutation = X[[omega_score_col]].copy()
    
    # For XGBoost, perform label encoding since it requires numeric labels
    if model_type == 'XGB':
        # Get unique cancer types and create a mapping
        print("Encoding cancer type labels for XGBoost...")
        unique_classes = np.unique(y)
        class_to_index = {cls: i for i, cls in enumerate(unique_classes)}
        index_to_class = {i: cls for cls, i in class_to_index.items()}
        
        # Convert string labels to numeric
        y_encoded = np.array([class_to_index[label] for label in y])
        
        # Debug
        print(f"Original classes: {unique_classes}")
        print(f"Encoded classes: {list(range(len(unique_classes)))}")
        print(f"Sample encoded values: {y_encoded[:5] if len(y_encoded) >= 5 else y_encoded}")
        
        # Use encoded labels instead of original
        y_for_training = y_encoded
        classes = unique_classes
    else:
        # Random Forest can handle string labels directly
        y_for_training = y
        classes = np.unique(y)
    
    # Parameter grid based on model type
    if model_type == 'RF':
        param_grid = {
            'n_estimators': [10, 15, 20, 25, 30, 35, 40, 45, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8, 16]
        }
        base_model = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    elif model_type == 'XGB':
        param_grid = {
            'n_estimators': [25, 50, 75, 100],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'max_depth': [5, 7, 9],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        base_model = XGBClassifier(random_state=random_state, eval_metric='mlogloss')
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'RF' or 'XGB'.")
    
    # Create outer folds
    if clinical_df is not None:
        print("Creating stratified folds for tissue localization...")
        fold_assignments = stratified_fold_assignment(clinical_df, n_splits=outer_splits)
        unique_indices = X.index.unique()
        fold_indices = []
        
        for fold_id in range(outer_splits):
            fold_patients = fold_assignments[fold_assignments == fold_id].index
            fold_idx = [i for i, idx in enumerate(X.index) 
                            if idx in fold_patients]
            # Only add non-empty fold indices
            if len(fold_idx) > 0:
                fold_indices.append(fold_idx)
        
        # Make sure we have at least one valid train/test split
        if len(fold_indices) >= 2:
            # Create train/test splits from the fold indices
            train_test_splits = []
            
            for i in range(len(fold_indices)):
                test_idx = fold_indices[i]
                train_idx = [idx for j, fold in enumerate(fold_indices) for idx in fold if j != i]
                train_test_splits.append((train_idx, test_idx))
            
            outer_cv = KFold(n_splits=len(train_test_splits), shuffle=False)
            outer_cv.split = lambda X, y=None, groups=None: iter(train_test_splits)
        else:
            # Not enough valid folds, fall back to StratifiedKFold
            print("Warning: Not enough valid folds from clinical data, falling back to StratifiedKFold.")
            outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    else:
        # Use regular stratified K-fold if no clinical data is provided
        outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    
    # Initialize result containers
    fold_results = []
    all_predictions = np.zeros_like(y, dtype=object)
    all_predictions.fill('')  # Initialize with empty strings
    
    # Confusion matrices for each fold
    confusion_matrices = []
    feature_importances = []
    
    # Prepare splits for tqdm
    splits = list(outer_cv.split(X.values, y_for_training))  # Use encoded labels for stratification if XGBoost
    
    # Perform nested cross-validation
    for fold, (train_idx, test_idx) in enumerate(tqdm(splits, desc=f"{model_type} tissue localization CV folds")):
        # Get original data for this fold
        X_train_orig = X.iloc[train_idx].copy()
        X_test_orig = X.iloc[test_idx].copy()
        
        if model_type == 'XGB':
            # Use encoded labels for XGBoost
            y_train = y_for_training[train_idx]
            y_test = y_for_training[test_idx]
            # Keep original labels for evaluation
            y_test_original = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
        else:
            y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            y_test = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
            y_test_original = y_test  # No encoding for RF
        
        # Normalize protein data - prevents data leakage by using only training data for normalization parameters
        X_protein_train = X_protein.iloc[train_idx].copy()
        X_protein_test = X_protein.iloc[test_idx].copy()
        
        # Use numerical indices for normalization to avoid KeyError
        # Create index mapping for both train and test sets
        train_indices = list(range(len(X_protein_train)))
        
        # Normalize protein data using numerical indices
        X_protein_train_norm, percentile_values = normalize_protein_data(
            X_protein_train.reset_index(drop=True), clinical_df, training_indices=train_indices)
        
        # Apply same normalization to test data using parameters from training
        X_protein_test_norm, _ = normalize_protein_data(
            X_protein_test.reset_index(drop=True), clinical_df, training_indices=None, percentile_values=percentile_values)
        
        # Normalize mutation data if available
        if has_mutation_data:
            X_mutation_train = X_mutation.iloc[train_idx].copy()
            X_mutation_test = X_mutation.iloc[test_idx].copy()
            
            # Normalize mutation data using numerical indices
            X_mutation_train_norm, detection_limits = normalize_mutation_data(
                X_mutation_train.reset_index(drop=True), clinical_df, training_indices=train_indices)
            
            # Apply same normalization to test data using parameters from training
            X_mutation_test_norm, _ = normalize_mutation_data(
                X_mutation_test.reset_index(drop=True), clinical_df, training_indices=None, stored_limits=detection_limits)
            
            # Combine protein and mutation features
            X_train_combined = pd.concat([X_protein_train_norm, X_mutation_train_norm], axis=1)
            X_test_combined = pd.concat([X_protein_test_norm, X_mutation_test_norm], axis=1)
        else:
            # Only use protein features
            X_train_combined = X_protein_train_norm
            X_test_combined = X_protein_test_norm
        
        # Create continuous features dataframe using normalized data
        X_train_continuous = X_train_combined[features].copy()
        X_test_continuous = X_test_combined[features].copy()
        
        # Apply log transformation to protein features if specified
        if log_transform in ['log2', 'log10']:
            for feature in protein_features:
                if feature in X_train_continuous.columns:
                    # Add a small constant to avoid log(0)
                    if log_transform == 'log2':
                        X_train_continuous[feature] = np.log2(X_train_continuous[feature] + 1e-10)
                        X_test_continuous[feature] = np.log2(X_test_continuous[feature] + 1e-10)
                    elif log_transform == 'log10':
                        X_train_continuous[feature] = np.log10(X_train_continuous[feature] + 1e-10)
                        X_test_continuous[feature] = np.log10(X_test_continuous[feature] + 1e-10)
        
        # Standardize continuous features if requested
        if standardize_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_continuous)
            X_test_scaled = scaler.transform(X_test_continuous)
        else:
            X_train_scaled = X_train_continuous.values
            X_test_scaled = X_test_continuous.values
        
        # Handle gender if requested
        if include_gender and has_gender:
            # Create one-hot encoder for gender
            gender_encoder = OneHotEncoder(sparse=False, drop='first')
            gender_train_data = gender_encoder.fit_transform(X_train_orig[['Sex']])
            gender_test_data = gender_encoder.transform(X_test_orig[['Sex']])
            
            # Combine with scaled features
            X_train_processed = np.hstack((X_train_scaled, gender_train_data))
            X_test_processed = np.hstack((X_test_scaled, gender_test_data))
            
            # Update feature names
            feature_names = features + [f"{gender_col}_encoded"]
        else:
            X_train_processed = X_train_scaled
            X_test_processed = X_test_scaled
            feature_names = features
        
        # Inner cross-validation for hyperparameter optimization
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        
        # Setup grid search with inner CV
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0,
            error_score='raise'  # Add this to debug potential errors
        )
        
        try:
            # Find best hyperparameters
            grid_search.fit(X_train_processed, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Make predictions on test set
            if model_type == 'XGB':
                # For XGBoost, we predict numeric indices
                y_pred_indices = best_model.predict(X_test_processed)
                # Convert back to original cancer type strings
                y_pred = np.array([index_to_class[idx] for idx in y_pred_indices])
            else:
                # For RF, we get string predictions directly
                y_pred = best_model.predict(X_test_processed)
            
            # Store predictions
            for i, idx in enumerate(test_idx):
                all_predictions[idx] = y_pred[i]
            
            # Calculate performance metrics
            fold_acc = accuracy_score(y_test_original, y_pred)
            cm = confusion_matrix(y_test_original, y_pred, labels=classes)
            
            # Store confusion matrix
            confusion_matrices.append({
                'fold': fold,
                'cm': cm,
                'classes': classes
            })
            
            # Store feature importance
            feature_importances.append({
                'fold': fold,
                'importances': best_model.feature_importances_,
                'feature_names': feature_names
            })
            
            # Store fold results
            fold_results.append({
                'fold': fold,
                'accuracy': fold_acc,
                'best_params': grid_search.best_params_,
                'test_indices': test_idx
            })
        except Exception as e:
            print(f"Error in fold {fold}: {str(e)}")
            print(f"X_train_processed shape: {X_train_processed.shape}")
            print(f"y_train shape: {y_train.shape if hasattr(y_train, 'shape') else len(y_train)}")
            print(f"y_train values: {np.unique(y_train)}")
            continue
    
    # Calculate overall metrics
    mask = all_predictions != ''
    if np.any(mask):
        overall_acc = accuracy_score(y[mask], all_predictions[mask])
        overall_cm = confusion_matrix(y[mask], all_predictions[mask], labels=classes)
    else:
        overall_acc = np.nan
        overall_cm = np.zeros((len(classes), len(classes)))
    
    # Aggregate feature importances
    if feature_importances:
        avg_importances = np.zeros(len(feature_names))
        for fi in feature_importances:
            avg_importances += fi['importances']
        avg_importances /= len(feature_importances)
    else:
        avg_importances = np.array([])
    
    print(f"Completed tissue localization model with overall accuracy: {overall_acc:.4f}")
    
    return {
        'fold_results': fold_results,
        'overall_accuracy': overall_acc,
        'predictions': all_predictions,
        'confusion_matrices': confusion_matrices,
        'overall_confusion_matrix': overall_cm,
        'classes': classes,
        'feature_importances': {
            'avg_importances': avg_importances,
            'feature_names': feature_names,
            'fold_importances': feature_importances
        }
    }

def combined_cancer_detection_and_localization(X, y_cancer_status, y_cancer_type, clinical_df=None,
                                               outer_splits=10, inner_splits=5, random_state=42,
                                               standardize_features=True, log_transform=None,
                                               detection_model='LR', localization_model='RF'):
    """
    Perform combined cancer detection and tissue localization with nested cross-validation
    for proper hyperparameter optimization of both models.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame containing features for model
    y_cancer_status : pandas.Series or numpy.ndarray
        Binary target variable (cancer vs normal)
    y_cancer_type : pandas.Series or numpy.ndarray
        Cancer type target variable (for cancer samples only)
    clinical_df : pandas.DataFrame, optional
        DataFrame with clinical information for stratified sampling
    outer_splits : int, optional
        Number of folds for outer cross-validation (default: 10)
    inner_splits : int, optional
        Number of folds for inner cross-validation (default: 5)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    standardize_features : bool, optional
        Whether to standardize features (default: True)
    log_transform : str, optional
        Type of log transformation to apply to protein features ('log2', 'log10', or None)
    detection_model : str, optional
        Type of model to use for cancer detection ('LR', 'XGB', 'TF', 'MOE', default: 'LR')
    localization_model : str, optional
        Type of model to use for tissue localization ('RF', 'XGB', 'TF', 'MOE', default: 'RF')
        
    Returns:
    --------
    dict
        Dictionary containing combined results with proper nested CV results
    """
    # Check if TensorFlow is available when TF or MOE model is requested
    if ((detection_model == 'TF' or detection_model == 'MOE' or 
         localization_model == 'TF' or localization_model == 'MOE') and 
        not TENSORFLOW_AVAILABLE):
        print("TensorFlow is not available but TF/MOE model type was requested.")
        print("To use TensorFlow models, install with: pip install tensorflow")
        
        # Fall back to traditional models
        if detection_model == 'TF' or detection_model == 'MOE':
            print("Falling back to LR for detection model")
            detection_model = 'LR'
        
        if localization_model == 'TF' or localization_model == 'MOE':
            print("Falling back to RF for localization model")
            localization_model = 'RF'
    
    # Proteins from the paper for LR model
    lr_proteins = ['CA-125', 'CA19-9', 'CEA', 'HGF', 'Myeloperoxidase', 'OPN', 'Prolactin', 'TIMP-1']
    
    # Create fold assignments for consistent partitioning
    if clinical_df is not None:
        fold_assignments = stratified_fold_assignment(clinical_df, n_splits=outer_splits)
    else:
        fold_assignments = None
    
    print(f"\n=== PHASE 1: Cancer Detection Model ({detection_model}) ===")
    # Run nested CV for cancer detection model
    if detection_model == 'TF':
        # Use TensorFlow fusion model
        lr_results = nested_cross_validation_tf(
            X, y_cancer_status, 
            protein_features=lr_proteins,
            omega_score_col='omega_score',
            clinical_df=clinical_df,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            random_state=random_state,
            standardize_features=standardize_features,
            log_transform=log_transform,
            is_multiclass=False
        )
    elif detection_model == 'MOE':
        # Use Mixture of Experts model
        lr_results = nested_cross_validation_moe(
            X, y_cancer_status, 
            protein_features=lr_proteins,
            omega_score_col='omega_score',
            clinical_df=clinical_df,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            random_state=random_state,
            standardize_features=standardize_features,
            log_transform=log_transform,
            is_multiclass=False
        )
    else:
        # Use traditional models (LR or XGB)
        lr_results = nested_cross_validation(
            X, y_cancer_status, 
            protein_features=lr_proteins,
            clinical_df=clinical_df,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            random_state=random_state,
            standardize_features=standardize_features,
            log_transform=log_transform,
            model_type=detection_model
        )
    
    # Extract cancer samples for tissue localization model (those correctly classified as cancer by detection model)
    cancer_mask = y_cancer_status == 1
    cancer_indices = np.where(cancer_mask)[0]
    correct_cancer_indices = cancer_indices[lr_results['predictions'][cancer_indices] == 1]
    
    print(f"\nDetected {len(correct_cancer_indices)} cancer samples out of {cancer_mask.sum()} true cancer samples")
    
    # Skip localization if no cancer samples were correctly classified
    if len(correct_cancer_indices) == 0:
        print("No cancer samples correctly detected. Skipping tissue localization.")
        return {
            'detection_results': lr_results,
            'localization_results': None,
            'detection_accuracy': (lr_results['predictions'] == y_cancer_status).mean(),
            'localization_accuracy': np.nan,
            'combined_accuracy': 0.0,
            'detection_correct': lr_results['predictions'] == y_cancer_status,
            'localization_correct': np.zeros_like(y_cancer_status, dtype=bool),
            'combined_correct': np.zeros_like(y_cancer_status, dtype=bool),
            'detection_model': detection_model,
            'localization_model': localization_model
        }
    
    print(f"\n=== PHASE 2: Tissue Localization Model ({localization_model}) ===")
    # Extract features and targets for correctly classified cancer samples
    X_cancer = X.iloc[correct_cancer_indices]
    y_cancer_type_filtered = y_cancer_type.iloc[correct_cancer_indices]
    
    # Create clinical data for cancer samples only if needed
    clinical_df_cancer = None
    if clinical_df is not None:
        cancer_patient_ids = X_cancer['Patient_ID'].unique()
        clinical_df_cancer = clinical_df[clinical_df['Patient_ID'].isin(cancer_patient_ids)]
    
    # Identify protein features for RF model (all proteins except those excluded)
    non_protein_cols = ['Patient_ID', 'Sample_ID', 'Tumor_type', 'AJCC_Stage', 
                         'CancerSEEK_Logistic_Regression_Score', 'CancerSEEK_Test_Result',
                         'is_cancer', 'Age', 'Race', 'Sex']
    omega_score_col = 'omega_score'
    protein_features = [col for col in X_cancer.columns if col not in non_protein_cols 
                       and col != omega_score_col]
    
    # Run nested CV for tissue localization model
    if localization_model == 'TF':
        # Use TensorFlow fusion model
        rf_results = nested_cross_validation_tf(
            X_cancer, 
            y_cancer_type_filtered,
            protein_features=protein_features,
            omega_score_col=omega_score_col,
            clinical_df=clinical_df_cancer,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            random_state=random_state,
            standardize_features=standardize_features,
            log_transform=log_transform,
            is_multiclass=True
        )
    elif localization_model == 'MOE':
        # Use Mixture of Experts model
        rf_results = nested_cross_validation_moe(
            X_cancer, 
            y_cancer_type_filtered,
            protein_features=protein_features,
            omega_score_col=omega_score_col,
            clinical_df=clinical_df_cancer,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            random_state=random_state,
            standardize_features=standardize_features,
            log_transform=log_transform,
            is_multiclass=True
        )
    else:
        # Use traditional models (RF or XGB)
        rf_results = nested_cross_validation_rf(
            X_cancer, 
            y_cancer_type_filtered,
            protein_features=protein_features,
            omega_score_col=omega_score_col,
            include_gender=True,
            clinical_df=clinical_df_cancer,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            random_state=random_state,
            standardize_features=standardize_features,
            log_transform=log_transform,
            model_type=localization_model
        )
    
    print("\n=== Combining results from both models ===")
    # Map RF predictions back to full dataset
    all_rf_predictions = np.empty_like(y_cancer_status, dtype=object)
    all_rf_predictions.fill('')
    
    for i, idx in enumerate(correct_cancer_indices):
        if i < len(rf_results['predictions']) and rf_results['predictions'][i] != '':
            all_rf_predictions[idx] = rf_results['predictions'][i]
    
    # Calculate overall combined performance
    # 1. Detection: Was cancer correctly detected?
    detection_correct = lr_results['predictions'] == y_cancer_status
    
    # 2. Localization: For true cancer patients correctly classified as cancer, was the cancer type correct?
    localization_correct = np.zeros_like(y_cancer_status, dtype=bool)
    
    for idx in correct_cancer_indices:
        pred_type = all_rf_predictions[idx]
        true_type = y_cancer_type[idx]
        localization_correct[idx] = (pred_type == true_type and pred_type != '')
    
    # 3. Combined: Was both detection and localization correct?
    combined_correct = detection_correct & localization_correct
    
    # Compute metrics
    detection_accuracy = detection_correct.mean()
    
    # Only consider cancer patients correctly classified as cancer for localization accuracy
    loc_accuracy_mask = np.zeros_like(y_cancer_status, dtype=bool)
    for idx in correct_cancer_indices:
        if all_rf_predictions[idx] != '':
            loc_accuracy_mask[idx] = True
    
    if np.any(loc_accuracy_mask):
        localization_accuracy = (all_rf_predictions[loc_accuracy_mask] == 
                               y_cancer_type[loc_accuracy_mask]).mean()
    else:
        localization_accuracy = np.nan
    
    combined_accuracy = combined_correct.mean()
    
    # Generate ROC curve data for detection model
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_cancer_status, lr_results['probabilities'])
    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'auc': lr_results['overall_auc']
    }
    
    print(f"Final results: Detection accuracy: {detection_accuracy:.4f}, Localization accuracy: {localization_accuracy:.4f}")
    
    return {
        'detection_results': lr_results,
        'localization_results': rf_results,
        'detection_accuracy': detection_accuracy,
        'localization_accuracy': localization_accuracy,
        'combined_accuracy': combined_accuracy,
        'detection_correct': detection_correct,
        'localization_correct': localization_correct,
        'combined_correct': combined_correct,
        'all_rf_predictions': all_rf_predictions,
        'roc_data': roc_data,
        'detection_model': detection_model,
        'localization_model': localization_model
    }

class TensorFlowFusionModel:
    """
    LATE fusion model using TensorFlow for combined feature processing.
    This model trains two separate models:
    1. A logistic regression for the omega_score
    2. A shallow neural network for protein features
    
    The model supports both binary classification (cancer detection) and
    multi-class classification (tissue localization).
    """
    
    def __init__(self, num_classes=2, hidden_units=32, dropout_rate=0.3, 
                 l1_reg=0.01, l2_reg=0.01, random_state=42):
        """
        Initialize the TensorFlow fusion model.
        
        Parameters:
        -----------
        num_classes : int, optional
            Number of output classes (default: 2 for binary classification)
        hidden_units : int, optional
            Number of neurons in the hidden layer (default: 32)
        dropout_rate : float, optional
            Dropout rate for regularization (default: 0.3)
        l1_reg : float, optional
            L1 regularization factor (default: 0.01)
        l2_reg : float, optional
            L2 regularization factor (default: 0.01)
        random_state : int, optional
            Random seed for reproducibility (default: 42)
        """
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.random_state = random_state
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def _build_model(self, protein_input_dim):
        """
        Build the fusion model architecture.
        
        Parameters:
        -----------
        protein_input_dim : int
            Number of protein features
            
        Returns:
        --------
        model : tf.keras.Model
            Compiled Keras model
        """
        # Set output activation and loss based on classification type
        if self.num_classes == 2:
            output_activation = 'sigmoid'
            loss = 'binary_crossentropy'
            output_units = 1
        else:
            output_activation = 'softmax'
            loss = 'categorical_crossentropy'
            output_units = self.num_classes
        
        # Input layers
        protein_input = layers.Input(shape=(protein_input_dim,), name='protein_input')
        omega_input = layers.Input(shape=(1,), name='omega_input')
        
        # Protein branch - shallow neural network
        protein_branch = layers.Dense(
            self.hidden_units, 
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='protein_dense'
        )(protein_input)
        protein_branch = layers.Dropout(self.dropout_rate, name='protein_dropout')(protein_branch)
        
        # Omega branch - direct connection (similar to logistic regression)
        omega_branch = layers.Dense(
            8, 
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='omega_dense'
        )(omega_input)
        
        # Concatenate both branches
        merged = layers.Concatenate(name='fusion')([protein_branch, omega_branch])
        
        # Output layer
        output = layers.Dense(
            output_units, 
            activation=output_activation,
            kernel_regularizer=regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='output'
        )(merged)
        
        # Create and compile model
        model = models.Model(inputs=[protein_input, omega_input], outputs=output)
        model.compile(
            optimizer=optimizers.Adam(),  # Default learning rate, will be tuned in fit
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X_protein, X_omega, y, validation_data=None, 
            learning_rates=[0.001, 0.01], 
            batch_sizes=[16, 32, 64], 
            epochs_list=[50, 100, 200],
            verbose=0):
        """
        Fit the model with grid search for hyperparameters.
        
        Parameters:
        -----------
        X_protein : array-like
            Protein features
        X_omega : array-like
            Omega score feature (mutation data)
        y : array-like
            Target variable
        validation_data : tuple, optional
            Validation data as ((X_protein_val, X_omega_val), y_val)
        learning_rates : list, optional
            List of learning rates to try (default: [0.001, 0.01])
        batch_sizes : list, optional
            List of batch sizes to try (default: [16, 32, 64])
        epochs_list : list, optional
            List of number of epochs to try (default: [50, 100, 200])
        verbose : int, optional
            Verbosity mode (default: 0)
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        # Check if y needs to be converted to categorical for multi-class
        if self.num_classes > 2:
            # Convert string labels to integers if necessary
            if isinstance(y[0], str):
                self.classes_ = np.unique(y)
                self.class_to_idx_ = {cls: i for i, cls in enumerate(self.classes_)}
                y_encoded = np.array([self.class_to_idx_[cls] for cls in y])
                y_categorical = utils.to_categorical(y_encoded, num_classes=self.num_classes)
            else:
                y_categorical = utils.to_categorical(y, num_classes=self.num_classes)
        else:
            y_categorical = y
        
        # Prepare validation data if provided
        if validation_data is not None:
            (X_protein_val, X_omega_val), y_val = validation_data
            if self.num_classes > 2 and isinstance(y_val[0], str):
                y_val_encoded = np.array([self.class_to_idx_[cls] for cls in y_val])
                y_val_categorical = utils.to_categorical(y_val_encoded, num_classes=self.num_classes)
            elif self.num_classes > 2:
                y_val_categorical = utils.to_categorical(y_val, num_classes=self.num_classes)
            else:
                y_val_categorical = y_val
            validation_data = ([X_protein_val, X_omega_val], y_val_categorical)
        
        # Reshape omega to ensure it's 2D
        if len(X_omega.shape) == 1:
            X_omega = X_omega.reshape(-1, 1)
        
        # Set up early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if validation_data is not None else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Grid search for hyperparameters
        best_val_loss = float('inf')
        best_params = {}
        best_model = None
        best_history = None
        
        # Iterate through all combinations
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for epochs in epochs_list:
                    if verbose > 0:
                        print(f"Trying lr={lr}, batch_size={batch_size}, epochs={epochs}")
                    
                    # Build model
                    model = self._build_model(X_protein.shape[1])
                    
                    # Set optimizer with current learning rate
                    model.optimizer = optimizers.Adam(learning_rate=lr)
                    
                    # Fit model
                    history = model.fit(
                        [X_protein, X_omega],
                        y_categorical,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=validation_data,
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    # Evaluate model
                    if validation_data is not None:
                        val_loss = min(history.history['val_loss'])
                    else:
                        # Use final training loss if no validation data
                        val_loss = history.history['loss'][-1]
                    
                    # Update best model if this one is better
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {'lr': lr, 'batch_size': batch_size, 'epochs': epochs}
                        best_model = model
                        best_history = history
        
        # Set the best model and parameters
        self.model = best_model
        self.best_params_ = best_params
        self.history = best_history
        
        if verbose > 0:
            print(f"Best parameters: {best_params}")
        
        return self
    
    def predict(self, X_protein, X_omega):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X_protein : array-like
            Protein features
        X_omega : array-like
            Omega score feature
            
        Returns:
        --------
        y_pred : array-like
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Reshape omega to ensure it's 2D
        if len(X_omega.shape) == 1:
            X_omega = X_omega.reshape(-1, 1)
        
        # Get predictions
        preds = self.model.predict([X_protein, X_omega], verbose=0)
        
        # Convert to class labels
        if self.num_classes > 2:
            # Get class indices with highest probability
            y_pred_indices = np.argmax(preds, axis=1)
            # Convert indices back to original class labels if needed
            if hasattr(self, 'classes_'):
                return np.array([self.classes_[idx] for idx in y_pred_indices])
            return y_pred_indices
        else:
            # Binary classification - threshold at 0.5
            return (preds > 0.5).astype(int).flatten()
    
    def predict_proba(self, X_protein, X_omega):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X_protein : array-like
            Protein features
        X_omega : array-like
            Omega score feature
            
        Returns:
        --------
        y_proba : array-like
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Reshape omega to ensure it's 2D
        if len(X_omega.shape) == 1:
            X_omega = X_omega.reshape(-1, 1)
        
        # Get predictions
        preds = self.model.predict([X_protein, X_omega], verbose=0)
        
        # For binary classification, return probability of positive class
        if self.num_classes == 2:
            return preds.flatten()
        
        # For multi-class, return probabilities for each class
        return preds

def nested_cross_validation_tf(X, y, protein_features, omega_score_col='omega_score', 
                            clinical_df=None, outer_splits=10, inner_splits=5, random_state=42,
                            standardize_features=True, log_transform=None, is_multiclass=False):
    """
    Perform nested cross-validation using the TensorFlow fusion model.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame containing features for model
    y : pandas.Series or numpy.ndarray
        Target variable (cancer vs normal, or cancer types)
    protein_features : list
        List of protein features to use in the model
    omega_score_col : str, optional
        Name of the omega score column (default: 'omega_score')
    clinical_df : pandas.DataFrame, optional
        DataFrame with clinical information for stratified sampling
    outer_splits : int, optional
        Number of folds for outer cross-validation (default: 10)
    inner_splits : int, optional
        Number of folds for inner cross-validation (default: 5)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    standardize_features : bool, optional
        Whether to standardize features (default: True)
    log_transform : str, optional
        Type of log transformation to apply to protein features ('log2', 'log10', or None)
    is_multiclass : bool, optional
        Whether this is a multi-class classification task (default: False)
        
    Returns:
    --------
    dict
        Dictionary containing nested cross-validation results
    """
    # Check if TensorFlow is available
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow is not available. Falling back to traditional models.")
        print("To use TensorFlow, install with: pip install tensorflow")
        
        # Fall back to traditional models
        if is_multiclass:
            print("Falling back to Random Forest for multi-class classification")
            return nested_cross_validation_rf(
                X, y, 
                protein_features=protein_features,
                omega_score_col=omega_score_col,
                clinical_df=clinical_df,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                random_state=random_state,
                standardize_features=standardize_features,
                log_transform=log_transform,
                model_type='RF'
            )
        else:
            print("Falling back to Logistic Regression for binary classification")
            return nested_cross_validation(
                X, y, 
                protein_features=protein_features,
                omega_score_col=omega_score_col,
                clinical_df=clinical_df,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                random_state=random_state,
                standardize_features=standardize_features,
                log_transform=log_transform,
                model_type='LR'
            )
    
    # If TensorFlow is available, continue with the original function
    print(f"Starting nested cross-validation for TensorFlow fusion model...")
    
    # Check if omega_score exists
    has_omega_score = omega_score_col in X.columns
    
    if not has_omega_score:
        print(f"Warning: {omega_score_col} not found in data. Adding dummy column with zeros.")
        X[omega_score_col] = 0.0
    
    # Determine number of classes for multi-class classification
    if is_multiclass:
        num_classes = len(np.unique(y))
        print(f"Multi-class classification with {num_classes} classes")
    else:
        num_classes = 2
        print("Binary classification")
    
    # Create outer folds
    if clinical_df is not None:
        print("Creating stratified folds based on clinical data...")
        fold_assignments = stratified_fold_assignment(clinical_df, n_splits=outer_splits)
        unique_indices = X.index.unique()
        fold_indices = []
        
        for fold_id in range(outer_splits):
            fold_patients = fold_assignments[fold_assignments == fold_id].index
            fold_idx = [i for i, idx in enumerate(X.index) 
                            if idx in fold_patients]
            # Only add non-empty fold indices
            if len(fold_idx) > 0:
                fold_indices.append(fold_idx)
        
        # Make sure we have at least one valid train/test split
        if len(fold_indices) >= 2:
            # Create train/test splits from the fold indices
            train_test_splits = []
            
            for i in range(len(fold_indices)):
                test_idx = fold_indices[i]
                train_idx = [idx for j, fold in enumerate(fold_indices) for idx in fold if j != i]
                train_test_splits.append((train_idx, test_idx))
            
            outer_cv = KFold(n_splits=len(train_test_splits), shuffle=False)
            outer_cv.split = lambda X, y=None, groups=None: iter(train_test_splits)
        else:
            # Not enough valid folds, fall back to StratifiedKFold
            print("Warning: Not enough valid folds from clinical data, falling back to StratifiedKFold.")
            outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    else:
        outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    
    # Initialize result containers
    fold_results = []
    all_predictions = np.zeros_like(y, dtype=float if not is_multiclass else object)
    if is_multiclass:
        all_predictions = np.array(['' for _ in range(len(y))])
    
    all_probabilities = []
    if not is_multiclass:
        all_probabilities = np.zeros_like(y, dtype=float)
    
    # For ROC curve plotting (binary classification only)
    fpr_list = []
    tpr_list = []
    
    # Prepare splits for tqdm
    splits = list(outer_cv.split(X.values, y))
    
    # Separate protein data and mutation data
    X_protein = X[protein_features]
    X_omega = X[[omega_score_col]]
    
    # Perform nested cross-validation
    for fold, (train_idx, test_idx) in enumerate(tqdm(splits, desc=f"TF fusion model CV folds")):
        # Access the original data for proper normalization
        X_train_orig = X.iloc[train_idx].copy()
        X_test_orig = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
        y_test = y.iloc[test_idx] if isinstance(y, pd.Series) else y[test_idx]
        
        # Normalize protein data
        X_protein_train = X_protein.iloc[train_idx].copy()
        X_protein_test = X_protein.iloc[test_idx].copy()
        
        # Use numerical indices for normalization to avoid KeyError
        train_indices = list(range(len(X_protein_train)))
        
        # Normalize protein data using numerical indices
        X_protein_train_norm, percentile_values = normalize_protein_data(
            X_protein_train.reset_index(drop=True), clinical_df, training_indices=train_indices)
        
        # Apply same normalization to test data using parameters from training
        X_protein_test_norm, _ = normalize_protein_data(
            X_protein_test.reset_index(drop=True), clinical_df, training_indices=None, percentile_values=percentile_values)
        
        # Normalize mutation data
        X_omega_train = X_omega.iloc[train_idx].copy()
        X_omega_test = X_omega.iloc[test_idx].copy()
        
        # Normalize mutation data using numerical indices
        X_omega_train_norm, detection_limits = normalize_mutation_data(
            X_omega_train.reset_index(drop=True), clinical_df, training_indices=train_indices)
        
        # Apply same normalization to test data using parameters from training
        X_omega_test_norm, _ = normalize_mutation_data(
            X_omega_test.reset_index(drop=True), clinical_df, training_indices=None, stored_limits=detection_limits)
        
        # Apply log transformation to protein features if specified
        if log_transform in ['log2', 'log10']:
            for feature in protein_features:
                if feature in X_protein_train_norm.columns:
                    # Add a small constant to avoid log(0)
                    if log_transform == 'log2':
                        X_protein_train_norm[feature] = np.log2(X_protein_train_norm[feature] + 1e-10)
                        X_protein_test_norm[feature] = np.log2(X_protein_test_norm[feature] + 1e-10)
                    elif log_transform == 'log10':
                        X_protein_train_norm[feature] = np.log10(X_protein_train_norm[feature] + 1e-10)
                        X_protein_test_norm[feature] = np.log10(X_protein_test_norm[feature] + 1e-10)
        
        # Standardize features if requested
        if standardize_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_protein_train_norm)
            X_test_scaled = scaler.transform(X_protein_test_norm)
        else:
            X_train_scaled = X_protein_train_norm.values
            X_test_scaled = X_protein_test_norm.values
        
        # Extract omega score as a separate feature
        X_omega_train_scaled = X_omega_train_norm[omega_score_col].values
        X_omega_test_scaled = X_omega_test_norm[omega_score_col].values
        
        # Inner cross-validation for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        
        # Set up inner CV folds
        inner_fold_splits = list(inner_cv.split(X_train_scaled, y_train))
        
        # Train the model on the full training set after finding the best hyperparameters
        # using a subset of the training data for validation
        # This simulates how the inner grid search would work
        
        # Split training data for validation
        inner_train_idx, inner_val_idx = inner_fold_splits[0]  # Use first split for validation
        
        # Create validation set
        X_protein_val = X_train_scaled[inner_val_idx]
        X_omega_val = X_omega_train_scaled[inner_val_idx]
        y_val = y_train[inner_val_idx] if not isinstance(y_train, pd.Series) else y_train.iloc[inner_val_idx].values
        
        # Reduce training set
        X_protein_inner_train = X_train_scaled[inner_train_idx]
        X_omega_inner_train = X_omega_train_scaled[inner_train_idx]
        y_inner_train = y_train[inner_train_idx] if not isinstance(y_train, pd.Series) else y_train.iloc[inner_train_idx].values
        
        # Initialize and train model with hyperparameter search
        tf_model = TensorFlowFusionModel(
            num_classes=num_classes,
            hidden_units=32,
            dropout_rate=0.3,
            l1_reg=0.01,
            l2_reg=0.01,
            random_state=random_state
        )
        
        # Only do grid search on first fold to save time
        learning_rates = [0.001, 0.01] if fold == 0 else [0.001]
        batch_sizes = [16, 32, 64] if fold == 0 else [32]
        epochs_list = [50, 100] if fold == 0 else [50]
        
        tf_model.fit(
            X_protein_inner_train, 
            X_omega_inner_train,
            y_inner_train,
            validation_data=((X_protein_val, X_omega_val), y_val),
            learning_rates=learning_rates,
            batch_sizes=batch_sizes,
            epochs_list=epochs_list,
            verbose=1 if fold == 0 else 0
        )
        
        # Now retrain on the full training set using the best hyperparameters
        print(f"Fold {fold}: Retraining with best parameters: {tf_model.best_params_}")
        
        # Initialize final model with the best parameters
        final_model = TensorFlowFusionModel(
            num_classes=num_classes,
            hidden_units=32,
            dropout_rate=0.3,
            l1_reg=0.01,
            l2_reg=0.01,
            random_state=random_state
        )
        
        # Fit final model on all training data
        final_model.fit(
            X_train_scaled,
            X_omega_train_scaled,
            y_train,
            validation_data=None,  # No validation set for final training
            learning_rates=[tf_model.best_params_['lr']],
            batch_sizes=[tf_model.best_params_['batch_size']],
            epochs_list=[tf_model.best_params_['epochs']],
            verbose=0
        )
        
        # Make predictions on test set
        if is_multiclass:
            y_pred = final_model.predict(X_test_scaled, X_omega_test_scaled)
            # For ROC curves in binary case only
            if num_classes == 2:
                y_prob = final_model.predict_proba(X_test_scaled, X_omega_test_scaled)
            else:
                y_prob = None
        else:
            y_pred = final_model.predict(X_test_scaled, X_omega_test_scaled)
            y_prob = final_model.predict_proba(X_test_scaled, X_omega_test_scaled)
        
        # Store predictions
        if is_multiclass:
            for i, idx in enumerate(test_idx):
                all_predictions[idx] = y_pred[i]
        else:
            all_predictions[test_idx] = y_pred
            all_probabilities[test_idx] = y_prob
        
        # Calculate performance metrics
        if is_multiclass:
            fold_acc = accuracy_score(y_test, y_pred)
            fold_results.append({
                'fold': fold,
                'accuracy': fold_acc,
                'best_params': tf_model.best_params_,
                'test_indices': test_idx
            })
        else:
            fold_acc = accuracy_score(y_test, y_pred)
            fold_auc = roc_auc_score(y_test, y_prob)
            
            # Calculate ROC curve data for this fold
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            
            fold_results.append({
                'fold': fold,
                'accuracy': fold_acc,
                'auc': fold_auc,
                'best_params': tf_model.best_params_,
                'test_indices': test_idx
            })
    
    # Calculate overall metrics
    if is_multiclass:
        mask = all_predictions != ''
        if np.any(mask):
            overall_acc = accuracy_score(y[mask], all_predictions[mask])
        else:
            overall_acc = np.nan
            
        print(f"Completed TF fusion model with overall accuracy: {overall_acc:.4f}")
        
        return {
            'fold_results': fold_results,
            'overall_accuracy': overall_acc,
            'predictions': all_predictions
        }
    else:
        overall_acc = accuracy_score(y, all_predictions)
        overall_auc = roc_auc_score(y, all_probabilities)
        
        # Calculate mean ROC curve for plotting
        from sklearn.metrics import roc_curve
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        
        for i in range(len(fpr_list)):
            mean_tpr += np.interp(mean_fpr, fpr_list[i], tpr_list[i])
        
        mean_tpr /= len(fpr_list)
        
        print(f"Completed TF fusion model with overall AUC: {overall_auc:.4f}")
        
        return {
            'fold_results': fold_results,
            'overall_accuracy': overall_acc,
            'overall_auc': overall_auc,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'roc_data': {
                'mean_fpr': mean_fpr,
                'mean_tpr': mean_tpr,
                'fpr_list': fpr_list,
                'tpr_list': tpr_list
            }
        }

class MixtureOfExpertsModel:
    """
    Mixture of Experts model using TensorFlow with dynamic routing.
    
    This model consists of:
    1. A gating network that decides which expert(s) to use for each input
    2. 8 expert networks with independent parameters
    3. A final layer that combines outputs from selected experts
    
    The model supports both binary classification (cancer detection) and
    multi-class classification (tissue localization).
    """
    
    def __init__(self, num_classes=2, num_experts=8, expert_units=32, 
                 gating_units=16, dropout_rate=0.3, 
                 l1_reg=0.01, l2_reg=0.01, random_state=42):
        """
        Initialize the Mixture of Experts model.
        
        Parameters:
        -----------
        num_classes : int, optional
            Number of output classes (default: 2 for binary classification)
        num_experts : int, optional
            Number of expert networks (default: 8)
        expert_units : int, optional
            Number of neurons in each expert's hidden layer (default: 32)
        gating_units : int, optional
            Number of neurons in the gating network (default: 16)
        dropout_rate : float, optional
            Dropout rate for regularization (default: 0.3)
        l1_reg : float, optional
            L1 regularization factor (default: 0.01)
        l2_reg : float, optional
            L2 regularization factor (default: 0.01)
        random_state : int, optional
            Random seed for reproducibility (default: 42)
        """
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.expert_units = expert_units
        self.gating_units = gating_units
        self.dropout_rate = dropout_rate
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.random_state = random_state
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def _build_model(self, protein_input_dim):
        """
        Build the Mixture of Experts model architecture.
        
        Parameters:
        -----------
        protein_input_dim : int
            Number of protein features
            
        Returns:
        --------
        model : tf.keras.Model
            Compiled Keras model
        """
        # Set output activation and loss based on classification type
        if self.num_classes == 2:
            output_activation = 'sigmoid'
            loss = 'binary_crossentropy'
            output_units = 1
        else:
            output_activation = 'softmax'
            loss = 'categorical_crossentropy'
            output_units = self.num_classes
        
        # Input layers
        protein_input = layers.Input(shape=(protein_input_dim,), name='protein_input')
        omega_input = layers.Input(shape=(1,), name='omega_input')
        
        # Combine inputs for full feature processing
        combined_input = layers.Concatenate(name='combined_input')([protein_input, omega_input])
        
        # Gating network
        gating = layers.Dense(
            self.gating_units, 
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='gating_dense_1'
        )(combined_input)
        gating = layers.Dropout(self.dropout_rate, name='gating_dropout')(gating)
        gating = layers.Dense(
            self.num_experts, 
            activation='softmax',
            kernel_regularizer=regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
            name='gating_output'
        )(gating)
        
        # Expert networks
        expert_outputs = []
        for i in range(self.num_experts):
            expert = layers.Dense(
                self.expert_units, 
                activation='relu',
                kernel_regularizer=regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name=f'expert_{i}_dense_1'
            )(combined_input)
            expert = layers.Dropout(self.dropout_rate, name=f'expert_{i}_dropout')(expert)
            expert = layers.Dense(
                output_units,
                activation=output_activation if i == self.num_experts - 1 else 'linear',
                kernel_regularizer=regularizers.l1_l2(l1=self.l1_reg, l2=self.l2_reg),
                name=f'expert_{i}_output'
            )(expert)
            expert_outputs.append(expert)
        
        # Stack expert outputs for weighted combination
        if self.num_experts > 1:
            stacked_experts = layers.Lambda(
                lambda x: tf.stack(x, axis=1),
                name='stack_experts'
            )(expert_outputs)
            
            # Add dimension to gating weights for proper broadcasting
            gating_weights = layers.Reshape((self.num_experts, 1), name='reshape_gating')(gating)
            
            # Multiply experts by gating weights
            weighted_experts = layers.Multiply(name='weight_experts')([stacked_experts, gating_weights])
            
            # Sum over experts dimension
            output = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1), name='combine_experts')(weighted_experts)
        else:
            # If only one expert, just use its output
            output = expert_outputs[0]
        
        # Create and compile model
        model = models.Model(inputs=[protein_input, omega_input], outputs=output)
        model.compile(
            optimizer=optimizers.Adam(),  # Default learning rate, will be tuned in fit
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X_protein, X_omega, y, validation_data=None, 
            learning_rates=[0.001, 0.0005, 0.0001], 
            batch_sizes=[16, 32, 64], 
            epochs_list=[50, 100, 200],
            verbose=0):
        """
        Fit the model with grid search for hyperparameters.
        
        Parameters:
        -----------
        X_protein : array-like
            Protein features
        X_omega : array-like
            Omega score feature (mutation data)
        y : array-like
            Target variable
        validation_data : tuple, optional
            Validation data as ((X_protein_val, X_omega_val), y_val)
        learning_rates : list, optional
            List of learning rates to try (default: [0.001, 0.0005, 0.0001])
        batch_sizes : list, optional
            List of batch sizes to try (default: [16, 32, 64])
        epochs_list : list, optional
            List of number of epochs to try (default: [50, 100, 200])
        verbose : int, optional
            Verbosity mode (default: 0)
            
        Returns:
        --------
        self : object
            Fitted estimator
        """
        # Check if y needs to be converted to categorical for multi-class
        if self.num_classes > 2:
            # Convert string labels to integers if necessary
            if isinstance(y[0], str):
                self.classes_ = np.unique(y)
                self.class_to_idx_ = {cls: i for i, cls in enumerate(self.classes_)}
                y_encoded = np.array([self.class_to_idx_[cls] for cls in y])
                y_categorical = utils.to_categorical(y_encoded, num_classes=self.num_classes)
            else:
                y_categorical = utils.to_categorical(y, num_classes=self.num_classes)
        else:
            y_categorical = y
        
        # Prepare validation data if provided
        if validation_data is not None:
            (X_protein_val, X_omega_val), y_val = validation_data
            if self.num_classes > 2 and isinstance(y_val[0], str):
                y_val_encoded = np.array([self.class_to_idx_[cls] for cls in y_val])
                y_val_categorical = utils.to_categorical(y_val_encoded, num_classes=self.num_classes)
            elif self.num_classes > 2:
                y_val_categorical = utils.to_categorical(y_val, num_classes=self.num_classes)
            else:
                y_val_categorical = y_val
            validation_data = ([X_protein_val, X_omega_val], y_val_categorical)
        
        # Reshape omega to ensure it's 2D
        if len(X_omega.shape) == 1:
            X_omega = X_omega.reshape(-1, 1)
        
        # Set up early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if validation_data is not None else 'loss',
            patience=15,
            restore_best_weights=True
        )
        
        # Add learning rate scheduler
        def lr_scheduler(epoch, lr):
            if epoch > 0 and epoch % 20 == 0:
                return lr * 0.9
            return lr
        
        lr_callback = callbacks.LearningRateScheduler(lr_scheduler)
        
        # Grid search for hyperparameters
        best_val_loss = float('inf')
        best_params = {}
        best_model = None
        best_history = None
        
        # Iterate through all combinations
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for epochs in epochs_list:
                    if verbose > 0:
                        print(f"Trying lr={lr}, batch_size={batch_size}, epochs={epochs}")
                    
                    # Build model
                    model = self._build_model(X_protein.shape[1])
                    
                    # Set optimizer with current learning rate
                    model.optimizer = optimizers.Adam(learning_rate=lr)
                    
                    # Fit model
                    history = model.fit(
                        [X_protein, X_omega],
                        y_categorical,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=validation_data,
                        callbacks=[early_stopping, lr_callback],
                        verbose=0
                    )
                    
                    # Evaluate model
                    if validation_data is not None:
                        val_loss = min(history.history['val_loss'])
                    else:
                        # Use final training loss if no validation data
                        val_loss = history.history['loss'][-1]
                    
                    # Update best model if this one is better
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_params = {'lr': lr, 'batch_size': batch_size, 'epochs': epochs}
                        best_model = model
                        best_history = history
        
        # Set the best model and parameters
        self.model = best_model
        self.best_params_ = best_params
        self.history = best_history
        
        if verbose > 0:
            print(f"Best parameters: {best_params}")
        
        return self
    
    def predict(self, X_protein, X_omega):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X_protein : array-like
            Protein features
        X_omega : array-like
            Omega score feature
            
        Returns:
        --------
        y_pred : array-like
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Reshape omega to ensure it's 2D
        if len(X_omega.shape) == 1:
            X_omega = X_omega.reshape(-1, 1)
        
        # Get predictions
        preds = self.model.predict([X_protein, X_omega], verbose=0)
        
        # Convert to class labels
        if self.num_classes > 2:
            # Get class indices with highest probability
            y_pred_indices = np.argmax(preds, axis=1)
            # Convert indices back to original class labels if needed
            if hasattr(self, 'classes_'):
                return np.array([self.classes_[idx] for idx in y_pred_indices])
            return y_pred_indices
        else:
            # Binary classification - threshold at 0.5
            return (preds > 0.5).astype(int).flatten()
    
    def predict_proba(self, X_protein, X_omega):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X_protein : array-like
            Protein features
        X_omega : array-like
            Omega score feature
            
        Returns:
        --------
        y_proba : array-like
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        
        # Reshape omega to ensure it's 2D
        if len(X_omega.shape) == 1:
            X_omega = X_omega.reshape(-1, 1)
        
        # Get predictions
        preds = self.model.predict([X_protein, X_omega], verbose=0)
        
        # For binary classification, return probability of positive class
        if self.num_classes == 2:
            return preds.flatten()
        
        # For multi-class, return probabilities for each class
        return preds

def nested_cross_validation_moe(X, y, protein_features, omega_score_col='omega_score', 
                            clinical_df=None, outer_splits=10, inner_splits=5, random_state=42,
                            standardize_features=True, log_transform=None, is_multiclass=False):
    """
    Perform nested cross-validation for Mixture of Experts model.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame containing features for model
    y : pandas.Series or numpy.ndarray
        Target variable (cancer vs normal for binary, or tumor types for multiclass)
    protein_features : list
        List of protein features to use in the model
    omega_score_col : str, optional
        Name of the omega score column (default: 'omega_score')
    clinical_df : pandas.DataFrame, optional
        DataFrame with clinical information for stratified sampling
    outer_splits : int, optional
        Number of folds for outer cross-validation (default: 10)
    inner_splits : int, optional
        Number of folds for inner cross-validation (default: 5)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    standardize_features : bool, optional
        Whether to standardize features (default: True)
    log_transform : str, optional
        Type of log transformation to apply to protein features ('log2', 'log10', or None)
    is_multiclass : bool, optional
        Whether this is a multiclass classification problem (default: False)
        
    Returns:
    --------
    dict
        Dictionary containing nested cross-validation results
    """
    # Check if TensorFlow is available
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow is not available. Falling back to traditional models.")
        print("To use TensorFlow, install with: pip install tensorflow")
        
        # Fall back to traditional models
        if is_multiclass:
            print("Falling back to Random Forest for multi-class classification")
            return nested_cross_validation_rf(
                X, y, 
                protein_features=protein_features,
                omega_score_col=omega_score_col,
                clinical_df=clinical_df,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                random_state=random_state,
                standardize_features=standardize_features,
                log_transform=log_transform,
                model_type='RF'
            )
        else:
            print("Falling back to Logistic Regression for binary classification")
            return nested_cross_validation(
                X, y, 
                protein_features=protein_features,
                omega_score_col=omega_score_col,
                clinical_df=clinical_df,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                random_state=random_state,
                standardize_features=standardize_features,
                log_transform=log_transform,
                model_type='LR'
            )
    
    # If TensorFlow is available, continue with the original function
    print(f"Starting nested cross-validation for Mixture of Experts model...")
    
    # Check if omega_score exists
    has_omega_score = omega_score_col in X.columns
    
    if not has_omega_score:
        print(f"Warning: {omega_score_col} not found in data. Adding dummy column with zeros.")
        X[omega_score_col] = 0.0
    
    # Convert y to numpy array if it's a pandas Series
    y_np = y.values if isinstance(y, pd.Series) else y
    
    # Determine number of classes for multi-class problems
    if is_multiclass:
        # Get unique classes
        classes = np.unique(y_np)
        num_classes = len(classes)
        print(f"Multi-class classification with {num_classes} classes")
        
        # Create a mapping from class labels to indices
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        # Initialize array for predictions (string type for class labels)
        all_predictions = np.array([''] * len(y_np), dtype=object)
    else:
        num_classes = 2
        print("Binary classification")
        
        # Initialize arrays for binary predictions
        all_predictions = np.zeros_like(y_np, dtype=float)
        all_probabilities = np.zeros_like(y_np, dtype=float)
    
    # Create outer folds
    if clinical_df is not None:
        print("Creating stratified folds based on clinical data...")
        fold_assignments = stratified_fold_assignment(clinical_df, n_splits=outer_splits)
        unique_indices = X.index.unique()
        fold_indices = []
        
        for fold_id in range(outer_splits):
            fold_patients = fold_assignments[fold_assignments == fold_id].index
            fold_idx = [i for i, idx in enumerate(X.index) 
                            if idx in fold_patients]
            # Only add non-empty fold indices
            if len(fold_idx) > 0:
                fold_indices.append(fold_idx)
        
        # Make sure we have at least one valid train/test split
        if len(fold_indices) >= 2:
            # Create train/test splits from the fold indices
            train_test_splits = []
            
            for i in range(len(fold_indices)):
                test_idx = fold_indices[i]
                train_idx = [idx for j, fold in enumerate(fold_indices) for idx in fold if j != i]
                train_test_splits.append((train_idx, test_idx))
            
            outer_cv = KFold(n_splits=len(train_test_splits), shuffle=False)
            outer_cv.split = lambda X, y=None, groups=None: iter(train_test_splits)
        else:
            # Not enough valid folds, fall back to StratifiedKFold
            print("Warning: Not enough valid folds from clinical data, falling back to StratifiedKFold.")
            outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    else:
        outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    
    # Initialize containers for results
    fold_results = []
    
    # For ROC curve plotting (binary classification only)
    if not is_multiclass:
        fpr_list = []
        tpr_list = []
    
    # Prepare splits for tqdm
    splits = list(outer_cv.split(X.values, y_np))
    
    # Separate protein data and mutation data
    protein_cols = protein_features
    X_protein = X[protein_cols]
    
    # Check if mutation data (omega score) exists
    X_mutation = X[[omega_score_col]]
    
    # Perform nested cross-validation
    for fold, (train_idx, test_idx) in enumerate(tqdm(splits, desc="MOE model CV folds")):
        # Access the original data for proper normalization
        X_train_orig = X.iloc[train_idx].copy()
        X_test_orig = X.iloc[test_idx].copy()
        
        # Get training and test labels
        if isinstance(y, pd.Series):
            y_train = y.iloc[train_idx].values
            y_test = y.iloc[test_idx].values
        else:
            y_train = y_np[train_idx]
            y_test = y_np[test_idx]
        
        # Normalize protein data - prevents data leakage by using only training data for normalization parameters
        X_protein_train = X_protein.iloc[train_idx].copy()
        X_protein_test = X_protein.iloc[test_idx].copy()
        
        # Use numerical indices for normalization to avoid KeyError
        # Create index mapping for train set
        train_indices = list(range(len(X_protein_train)))
        
        # Normalize protein data using numerical indices
        X_protein_train_norm, percentile_values = normalize_protein_data(
            X_protein_train.reset_index(drop=True), clinical_df, training_indices=train_indices)
        
        # Apply same normalization to test data using parameters from training
        X_protein_test_norm, _ = normalize_protein_data(
            X_protein_test.reset_index(drop=True), clinical_df, training_indices=None, percentile_values=percentile_values)
        
        # Normalize mutation data if available
        X_mutation_train = X_mutation.iloc[train_idx].copy()
        X_mutation_test = X_mutation.iloc[test_idx].copy()
        
        # Normalize mutation data using numerical indices
        X_mutation_train_norm, detection_limits = normalize_mutation_data(
            X_mutation_train.reset_index(drop=True), clinical_df, training_indices=train_indices)
        
        # Apply same normalization to test data using parameters from training
        X_mutation_test_norm, _ = normalize_mutation_data(
            X_mutation_test.reset_index(drop=True), clinical_df, training_indices=None, stored_limits=detection_limits)
        
        # Apply log transformation to protein features if specified
        if log_transform in ['log2', 'log10']:
            for feature in protein_cols:
                if feature in X_protein_train_norm.columns:
                    # Add a small constant to avoid log(0)
                    if log_transform == 'log2':
                        X_protein_train_norm[feature] = np.log2(X_protein_train_norm[feature] + 1e-10)
                        X_protein_test_norm[feature] = np.log2(X_protein_test_norm[feature] + 1e-10)
                    elif log_transform == 'log10':
                        X_protein_train_norm[feature] = np.log10(X_protein_train_norm[feature] + 1e-10)
                        X_protein_test_norm[feature] = np.log10(X_protein_test_norm[feature] + 1e-10)
        
        # Standardize protein features if requested
        if standardize_features:
            protein_scaler = StandardScaler()
            X_protein_train_scaled = protein_scaler.fit_transform(X_protein_train_norm)
            X_protein_test_scaled = protein_scaler.transform(X_protein_test_norm)
            
            # Standardize omega score as well (separately)
            omega_scaler = StandardScaler()
            X_omega_train_scaled = omega_scaler.fit_transform(X_mutation_train_norm)
            X_omega_test_scaled = omega_scaler.transform(X_mutation_test_norm)
        else:
            X_protein_train_scaled = X_protein_train_norm.values
            X_protein_test_scaled = X_protein_test_norm.values
            X_omega_train_scaled = X_mutation_train_norm.values
            X_omega_test_scaled = X_mutation_test_norm.values
        
        # Create inner validation set from training data
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        inner_splits_indices = list(inner_cv.split(X_protein_train_scaled, y_train))
        inner_train_idx, inner_val_idx = inner_splits_indices[0]  # Use first split for validation
        
        # Train set for inner validation - make sure to use integer indexing with numpy arrays
        X_inner_train_protein = X_protein_train_scaled[inner_train_idx]
        X_inner_train_omega = X_omega_train_scaled[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]  # Now y_train is a numpy array, so this works properly
        
        # Validation set for inner validation
        X_inner_val_protein = X_protein_train_scaled[inner_val_idx]
        X_inner_val_omega = X_omega_train_scaled[inner_val_idx]
        y_inner_val = y_train[inner_val_idx]  # Now y_train is a numpy array, so this works properly
        
        # Initialize MoE model with appropriate number of classes
        moe_model = MixtureOfExpertsModel(
            num_classes=num_classes if is_multiclass else 2,
            num_experts=8,  # 8 experts as requested
            expert_units=32, 
            gating_units=16,
            dropout_rate=0.3,
            l1_reg=0.01,
            l2_reg=0.01,
            random_state=random_state
        )
        
        # Fit model with grid search for hyperparameters, using validation data
        moe_model.fit(
            X_inner_train_protein,
            X_inner_train_omega,
            y_inner_train,
            validation_data=((X_inner_val_protein, X_inner_val_omega), y_inner_val),
            learning_rates=[0.001, 0.0005, 0.0001],  # Try different learning rates
            batch_sizes=[16, 32, 64],  # Try different batch sizes
            epochs_list=[50, 100, 200],  # Try different epoch counts
            verbose=0
        )
        
        # Now retrain on the full training set using the best hyperparameters
        print(f"Fold {fold}: Retraining with best parameters: {moe_model.best_params_}")
        
        # Initialize final model with the best parameters
        final_model = MixtureOfExpertsModel(
            num_classes=num_classes if is_multiclass else 2,
            num_experts=8,
            expert_units=32,
            gating_units=16,
            dropout_rate=0.3,
            l1_reg=0.01,
            l2_reg=0.01,
            random_state=random_state
        )
        
        # Fit final model on all training data
        final_model.fit(
            X_protein_train_scaled,
            X_omega_train_scaled,
            y_train,
            validation_data=None,  # No validation set for final training
            learning_rates=[moe_model.best_params_['lr']],
            batch_sizes=[moe_model.best_params_['batch_size']],
            epochs_list=[moe_model.best_params_['epochs']],
            verbose=0
        )
        
        # Make predictions on test set
        if is_multiclass:
            y_pred = final_model.predict(X_protein_test_scaled, X_omega_test_scaled)
            # For ROC curves in binary case only
            if num_classes == 2:
                y_prob = final_model.predict_proba(X_protein_test_scaled, X_omega_test_scaled)
            else:
                y_prob = None
        else:
            y_pred = final_model.predict(X_protein_test_scaled, X_omega_test_scaled)
            y_prob = final_model.predict_proba(X_protein_test_scaled, X_omega_test_scaled)
        
        # Store predictions
        if is_multiclass:
            for i, idx in enumerate(test_idx):
                all_predictions[idx] = y_pred[i]
        else:
            all_predictions[test_idx] = y_pred
            all_probabilities[test_idx] = y_prob
        
        # Calculate performance metrics
        if is_multiclass:
            fold_acc = accuracy_score(y_test, y_pred)
            fold_result = {
                'fold': fold,
                'accuracy': fold_acc,
                'best_params': moe_model.best_params_,
                'test_indices': test_idx.tolist()
            }
        else:
            fold_acc = accuracy_score(y_test, y_pred)
            fold_auc = roc_auc_score(y_test, y_prob)
            
            # Calculate ROC curve data for this fold
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            
            fold_result = {
                'fold': fold,
                'accuracy': fold_acc,
                'auc': fold_auc,
                'best_params': moe_model.best_params_,
                'test_indices': test_idx.tolist()
            }
        
        # Store fold results
        fold_results.append(fold_result)
    
    # Calculate overall metrics
    if is_multiclass:
        mask = all_predictions != ''
        if np.any(mask):
            overall_acc = accuracy_score(y_np[mask], all_predictions[mask])
        else:
            overall_acc = np.nan
            
        print(f"Completed MOE model with overall accuracy: {overall_acc:.4f}")
        
        return {
            'fold_results': fold_results,
            'overall_accuracy': overall_acc,
            'predictions': all_predictions
        }
    else:
        overall_acc = accuracy_score(y_np, all_predictions)
        overall_auc = roc_auc_score(y_np, all_probabilities)
        
        # Calculate mean ROC curve for plotting
        from sklearn.metrics import roc_curve
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        
        for i in range(len(fpr_list)):
            mean_tpr += np.interp(mean_fpr, fpr_list[i], tpr_list[i])
        
        mean_tpr /= len(fpr_list)
        
        print(f"Completed MOE model with overall AUC: {overall_auc:.4f}")
        
        return {
            'fold_results': fold_results,
            'overall_accuracy': overall_acc,
            'overall_auc': overall_auc,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'roc_data': {
                'mean_fpr': mean_fpr,
                'mean_tpr': mean_tpr,
                'fpr_list': fpr_list,
                'tpr_list': tpr_list
            }
        }

