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
    
    # Filter columns to only include numeric, non-boolean columns for normalization
    numeric_cols = []
    for col in normalized_df.columns:
        if pd.api.types.is_numeric_dtype(normalized_df[col]) and not pd.api.types.is_bool_dtype(normalized_df[col]):
            numeric_cols.append(col)
    
    # Exclude certain columns that aren't proteins
    non_protein_cols = ['Patient_ID', 'Sample_ID', 'Tumor_type', 'AJCC_Stage', 
                        'CancerSEEK_Logistic_Regression_Score', 'CancerSEEK_Test_Result', 
                        'is_cancer', 'Sex']
    protein_cols = [col for col in numeric_cols if col not in non_protein_cols]
    
    # If no protein columns, return original data and empty dict
    if not protein_cols:
        print("No valid numeric protein columns found for normalization.")
        return normalized_df, {}
    
    # Identify normal samples from clinical data
    normal_indices = []
    if clinical_df is not None and 'Tumor_type' in clinical_df.columns:
        normal_samples = clinical_df[clinical_df['Tumor_type'] == 'Normal']['Patient_ID'].tolist()
        if 'Patient_ID' in normalized_df.columns:
            normal_indices = normalized_df[normalized_df['Patient_ID'].isin(normal_samples)].index.tolist()
    
    # Use provided training indices if available
    if training_indices is None:
        training_indices = normal_indices
    
    # Initialize percentile_values_dict
    if percentile_values is None:
        percentile_values_dict = {}
    else:
        percentile_values_dict = percentile_values.copy()
    
    # Process each protein column
    for protein in protein_cols:
        try:
            # Get or calculate detection limits
            if detection_limits and protein in detection_limits:
                lower_limit, upper_limit = detection_limits[protein]
            else:
                lower_limit = normalized_df[protein].min()
                upper_limit = normalized_df[protein].max()
            
            # Apply clipping to detection limits
            normalized_df[protein] = normalized_df[protein].clip(lower=lower_limit, upper=upper_limit)
            
            # Calculate or use provided 95th percentile
            if protein not in percentile_values_dict:
                if training_indices and len(training_indices) > 0:
                    # Extract values for the specific protein and indices
                    protein_values = normalized_df.loc[training_indices, protein]
                    # Drop any NaN values
                    protein_values = protein_values.dropna()
                    
                    if not protein_values.empty:
                        # Calculate 95th percentile
                        percentile_values_dict[protein] = protein_values.quantile(0.95)
                    else:
                        # No valid values, use overall median
                        percentile_values_dict[protein] = normalized_df[protein].median()
                else:
                    # No training indices, use overall percentile
                    percentile_values_dict[protein] = normalized_df[protein].quantile(0.95)
            
            # Set values below 95th percentile to zero
            threshold = percentile_values_dict[protein]
            normalized_df[protein] = normalized_df[protein].apply(lambda x: 0 if x < threshold else x)
        
        except Exception as e:
            print(f"Error normalizing {protein}: {str(e)}. Keeping original values.")
    
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
    try:
        # Create a copy of the input data to avoid modifying the original
        normalized_df = mutation_df.copy()
        
        # Check if omega_score is present and numeric
        if 'omega_score' not in normalized_df.columns:
            print("Warning: 'omega_score' column not found. Returning original data.")
            return normalized_df, (0, 1)
        
        # Check if omega_score is a valid numeric type
        if not (pd.api.types.is_numeric_dtype(normalized_df['omega_score']) and 
                not pd.api.types.is_bool_dtype(normalized_df['omega_score'])):
            print("Warning: 'omega_score' column is not numeric or is boolean. Returning original data.")
            return normalized_df, (0, 1)
        
        # Identify normal samples from clinical data
        normal_indices = []
        if clinical_df is not None and 'Tumor_type' in clinical_df.columns:
            normal_samples = clinical_df[clinical_df['Tumor_type'] == 'Normal']['Patient_ID'].tolist()
            if 'Patient_ID' in normalized_df.columns:
                normal_indices = normalized_df[normalized_df['Patient_ID'].isin(normal_samples)].index.tolist()
        
        # Use provided training indices if available
        if training_indices is None:
            training_indices = normal_indices
        
        # Determine limits
        if stored_limits is not None:
            lower_limit, upper_limit = stored_limits
        elif detection_limits is not None:
            lower_limit, upper_limit = detection_limits
        else:
            # Calculate from data
            if training_indices and len(training_indices) > 0:
                # Extract values
                omega_values = normalized_df.loc[training_indices, 'omega_score'].dropna()
                if not omega_values.empty:
                    lower_limit = omega_values.min()
                    upper_limit = omega_values.max()
                else:
                    # No values in training set, use overall min/max
                    lower_limit = normalized_df['omega_score'].min()
                    upper_limit = normalized_df['omega_score'].max()
            else:
                # No training indices, use overall min/max
                lower_limit = normalized_df['omega_score'].min()
                upper_limit = normalized_df['omega_score'].max()
        
        # Apply clipping to detection limits
        normalized_df['omega_score'] = normalized_df['omega_score'].clip(lower=lower_limit, upper=upper_limit)
        
        return normalized_df, (lower_limit, upper_limit)
    
    except Exception as e:
        print(f"Error in normalize_mutation_data: {str(e)}. Returning original data.")
        return mutation_df, (0, 1)

# TensorFlow-based model implementations

def create_tf_late_fusion_model(protein_input_shape, mutation_input_shape, 
                             num_classes=2, dropout_rate=0.5, 
                             l2_reg=0.01, learning_rate=0.001):
    """
    Create a TensorFlow Late Fusion model that processes protein and mutation data separately
    and then combines them for final prediction.
    
    Parameters:
    -----------
    protein_input_shape : tuple
        Shape of protein input features (n_features,)
    mutation_input_shape : tuple
        Shape of mutation input features (n_features,)
    num_classes : int, optional
        Number of output classes (default: 2 for binary cancer detection)
    dropout_rate : float, optional
        Dropout rate to prevent overfitting (default: 0.5)
    l2_reg : float, optional
        L2 regularization strength (default: 0.01)
    learning_rate : float, optional
        Learning rate for Adam optimizer (default: 0.001)
        
    Returns:
    --------
    keras.Model
        Compiled TensorFlow model for late fusion
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for this model but is not available.")
    
    # Input layers
    protein_inputs = layers.Input(shape=protein_input_shape, name='protein_input')
    mutation_inputs = layers.Input(shape=mutation_input_shape, name='mutation_input')
    
    # Process protein data
    x_protein = layers.Dense(64, activation='relu', 
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name='protein_dense_1')(protein_inputs)
    x_protein = layers.BatchNormalization(name='protein_bn_1')(x_protein)
    x_protein = layers.Dropout(dropout_rate, name='protein_dropout_1')(x_protein)
    x_protein = layers.Dense(32, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg),
                         name='protein_dense_2')(x_protein)
    x_protein = layers.BatchNormalization(name='protein_bn_2')(x_protein)
    x_protein = layers.Dropout(dropout_rate, name='protein_dropout_2')(x_protein)
    
    # Process mutation data
    x_mutation = layers.Dense(16, activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg),
                          name='mutation_dense_1')(mutation_inputs)
    x_mutation = layers.BatchNormalization(name='mutation_bn_1')(x_mutation)
    x_mutation = layers.Dropout(dropout_rate, name='mutation_dropout_1')(x_mutation)
    
    # Concatenate protein and mutation features (late fusion)
    concatenated = layers.Concatenate(name='concatenate')([x_protein, x_mutation])
    
    # Final prediction layers
    x = layers.Dense(32, activation='relu',
                  kernel_regularizer=regularizers.l2(l2_reg),
                  name='combined_dense_1')(concatenated)
    x = layers.BatchNormalization(name='combined_bn_1')(x)
    x = layers.Dropout(dropout_rate, name='combined_dropout_1')(x)
    
    # Output layer
    if num_classes == 2:
        # Binary classification
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
    else:
        # Multi-class classification
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    # Create model
    model = models.Model(inputs=[protein_inputs, mutation_inputs], outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model

def create_tf_moe_model(protein_input_shape, mutation_input_shape,
                     num_classes=2, num_experts=3, dropout_rate=0.5,
                     l2_reg=0.01, learning_rate=0.001):
    """
    Create a TensorFlow Mixture of Experts model that uses gating to dynamically
    combine predictions from different expert networks.
    
    Parameters:
    -----------
    protein_input_shape : tuple
        Shape of protein input features (n_features,)
    mutation_input_shape : tuple
        Shape of mutation input features (n_features,)
    num_classes : int, optional
        Number of output classes (default: 2 for binary cancer detection)
    num_experts : int, optional
        Number of expert networks (default: 3)
    dropout_rate : float, optional
        Dropout rate to prevent overfitting (default: 0.5)
    l2_reg : float, optional
        L2 regularization strength (default: 0.01)
    learning_rate : float, optional
        Learning rate for Adam optimizer (default: 0.001)
        
    Returns:
    --------
    keras.Model
        Compiled TensorFlow model for mixture of experts
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for this model but is not available.")
    
    # Input layers
    protein_inputs = layers.Input(shape=protein_input_shape, name='protein_input')
    mutation_inputs = layers.Input(shape=mutation_input_shape, name='mutation_input')
    
    # Combined inputs for gating network
    combined_inputs = layers.Concatenate(name='combined_inputs')([protein_inputs, mutation_inputs])
    
    # Gating network
    gating = layers.Dense(64, activation='relu',
                       kernel_regularizer=regularizers.l2(l2_reg),
                       name='gating_dense_1')(combined_inputs)
    gating = layers.BatchNormalization(name='gating_bn_1')(gating)
    gating = layers.Dropout(dropout_rate, name='gating_dropout_1')(gating)
    gating = layers.Dense(num_experts, activation='softmax',
                       name='gating_output')(gating)
    
    # Expert networks
    expert_outputs = []
    for i in range(num_experts):
        # Process protein data
        x_protein = layers.Dense(64, activation='relu',
                              kernel_regularizer=regularizers.l2(l2_reg),
                              name=f'expert_{i}_protein_dense_1')(protein_inputs)
        x_protein = layers.BatchNormalization(name=f'expert_{i}_protein_bn_1')(x_protein)
        x_protein = layers.Dropout(dropout_rate, name=f'expert_{i}_protein_dropout_1')(x_protein)
        
        # Process mutation data
        x_mutation = layers.Dense(16, activation='relu',
                               kernel_regularizer=regularizers.l2(l2_reg),
                               name=f'expert_{i}_mutation_dense_1')(mutation_inputs)
        x_mutation = layers.BatchNormalization(name=f'expert_{i}_mutation_bn_1')(x_mutation)
        x_mutation = layers.Dropout(dropout_rate, name=f'expert_{i}_mutation_dropout_1')(x_mutation)
        
        # Combine protein and mutation features for this expert
        x_combined = layers.Concatenate(name=f'expert_{i}_concatenate')([x_protein, x_mutation])
        x_combined = layers.Dense(32, activation='relu',
                               kernel_regularizer=regularizers.l2(l2_reg),
                               name=f'expert_{i}_combined_dense_1')(x_combined)
        
        # Expert output
        if num_classes == 2:
            # Binary classification
            expert_output = layers.Dense(1, activation='sigmoid',
                                      name=f'expert_{i}_output')(x_combined)
        else:
            # Multi-class classification
            expert_output = layers.Dense(num_classes, activation='softmax',
                                      name=f'expert_{i}_output')(x_combined)
        
        expert_outputs.append(expert_output)
    
    # Stack expert outputs
    if num_classes == 2:
        # For binary classification, shape each output to [batch_size, 1]
        stacked_experts = layers.Lambda(lambda x: tf.stack(x, axis=1),
                                     name='stack_experts')(expert_outputs)
    else:
        # For multi-class, shape each output to [batch_size, num_classes]
        stacked_experts = layers.Lambda(lambda x: tf.stack(x, axis=1),
                                     name='stack_experts')(expert_outputs)
    
    # Reshape gating for broadcasting
    gating_reshaped = layers.Reshape((num_experts, 1), name='reshape_gating')(gating)
    
    # Multiply experts by gating weights and sum
    weighted_outputs = layers.Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1),
                                  name='weighted_outputs')([stacked_experts, gating_reshaped])
    
    # Create model
    model = models.Model(inputs=[protein_inputs, mutation_inputs], outputs=weighted_outputs)
    
    # Compile model
    if num_classes == 2:
        loss = 'binary_crossentropy'
        metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
    else:
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )
    
    return model

def tf_cross_validation(X, y, protein_features, omega_score_col='omega_score', 
                     clinical_df=None, outer_splits=10, inner_splits=5, 
                     random_state=42, standardize_features=True, 
                     log_transform=None, model_type='TF',
                     num_classes=2, batch_size=32, epochs=100,
                     patience=10, dropout_rate=0.5):
    """
    Perform TensorFlow-based cross-validation for either the Late Fusion or MOE model.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame containing features for model
    y : pandas.Series or numpy.ndarray
        Target variable (cancer vs normal or cancer types)
    protein_features : list
        List of protein features to use in the model
    omega_score_col : str, optional
        Name of the omega score column (default: 'omega_score')
    clinical_df : pandas.DataFrame, optional
        DataFrame with clinical information for stratified sampling
    outer_splits : int, optional
        Number of folds for outer cross-validation (default: 10)
    inner_splits : int, optional
        Number of folds for inner cross-validation for hyperparameter tuning (default: 5)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    standardize_features : bool, optional
        Whether to standardize features (default: True)
    log_transform : str, optional
        Type of log transformation to apply to protein features ('log2', 'log10', or None)
    model_type : str, optional
        Type of TensorFlow model to use ('TF' for Late Fusion, 'MOE' for Mixture of Experts)
    num_classes : int, optional
        Number of output classes (default: 2 for binary cancer detection)
    batch_size : int, optional
        Batch size for training (default: 32)
    epochs : int, optional
        Maximum number of epochs for training (default: 100)
    patience : int, optional
        Patience for early stopping (default: 10)
    dropout_rate : float, optional
        Dropout rate for regularization (default: 0.5)
        
    Returns:
    --------
    dict
        Dictionary containing cross-validation results
    """
    print(f"Starting TensorFlow {model_type} cross-validation for {'cancer detection' if num_classes == 2 else 'tissue localization'}...")
    
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required for this model but is not available.")
    
    # Include omega score with protein features
    features = protein_features.copy()
    if omega_score_col in X.columns:
        features.append(omega_score_col)
    
    # Determine hyperparameter grid for tuning
    param_grid = {
        'dropout_rate': [0.3, 0.5, 0.7],
        'learning_rate': [0.01, 0.001, 0.0001],
        'units_protein': [32, 64, 128],
        'units_mutation': [8, 16, 32],
        'l2_reg': [0.001, 0.01, 0.1]
    }
    
    if model_type == 'MOE':
        param_grid['num_experts'] = [2, 3, 5]
    
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
            if len(fold_idx) > 0:
                fold_indices.append(fold_idx)
        
        if len(fold_indices) >= 2:
            train_test_splits = []
            for i in range(len(fold_indices)):
                test_idx = fold_indices[i]
                train_idx = [idx for j, fold in enumerate(fold_indices) for idx in fold if j != i]
                train_test_splits.append((train_idx, test_idx))
            
            outer_cv = KFold(n_splits=len(train_test_splits), shuffle=False)
            outer_cv.split = lambda X, y=None, groups=None: iter(train_test_splits)
        else:
            print("Warning: Not enough valid folds, falling back to StratifiedKFold.")
            outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    else:
        outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    
    # Initialize result containers
    fold_results = []
    all_probabilities = np.zeros_like(y, dtype=float)
    if num_classes > 2:
        all_predictions = np.empty_like(y, dtype=object)
    else:
        all_predictions = np.zeros_like(y, dtype=int)
    
    # For ROC curve plotting (binary classification)
    fpr_list = []
    tpr_list = []
    
    # Prepare splits for tqdm
    splits = list(outer_cv.split(X.values, y))
    
    # Perform cross-validation
    for fold, (train_idx, test_idx) in enumerate(tqdm(splits, desc=f"{model_type} CV folds")):
        # Extract train/test sets
        X_train_orig = X.iloc[train_idx].copy()
        X_test_orig = X.iloc[test_idx].copy()
        y_train = y[train_idx] if isinstance(y, np.ndarray) else y.iloc[train_idx].values
        y_test = y[test_idx] if isinstance(y, np.ndarray) else y.iloc[test_idx].values
        
        # Extract protein features
        X_protein_train = X_train_orig[protein_features].copy()
        X_protein_test = X_test_orig[protein_features].copy()
        
        # Apply log transformation if specified
        if log_transform in ['log2', 'log10']:
            for feature in protein_features:
                if feature in X_protein_train.columns:
                    if log_transform == 'log2':
                        X_protein_train[feature] = np.log2(X_protein_train[feature] + 1e-10)
                        X_protein_test[feature] = np.log2(X_protein_test[feature] + 1e-10)
                    elif log_transform == 'log10':
                        X_protein_train[feature] = np.log10(X_protein_train[feature] + 1e-10)
                        X_protein_test[feature] = np.log10(X_protein_test[feature] + 1e-10)
        
        # Extract mutation data (omega score)
        if omega_score_col in X.columns:
            X_mutation_train = X_train_orig[[omega_score_col]].copy()
            X_mutation_test = X_test_orig[[omega_score_col]].copy()
        else:
            # Create dummy omega score column
            X_mutation_train = pd.DataFrame(index=X_protein_train.index, columns=[omega_score_col])
            X_mutation_test = pd.DataFrame(index=X_protein_test.index, columns=[omega_score_col])
            X_mutation_train[omega_score_col] = 0
            X_mutation_test[omega_score_col] = 0
            
        # Standardize features if requested
        if standardize_features:
            # Standardize protein features
            protein_scaler = StandardScaler()
            X_protein_train_scaled = protein_scaler.fit_transform(X_protein_train)
            X_protein_test_scaled = protein_scaler.transform(X_protein_test)
            
            # Standardize mutation features
            mutation_scaler = StandardScaler()
            X_mutation_train_scaled = mutation_scaler.fit_transform(X_mutation_train)
            X_mutation_test_scaled = mutation_scaler.transform(X_mutation_test)
        else:
            X_protein_train_scaled = X_protein_train.values
            X_protein_test_scaled = X_protein_test.values
            X_mutation_train_scaled = X_mutation_train.values
            X_mutation_test_scaled = X_mutation_test.values
        
        # Prepare validation set from training data
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        inner_split = list(inner_cv.split(X_protein_train_scaled, y_train))[0]  # Take first split for validation
        val_idx = inner_split[1]
        
        # Split training data into train and validation sets
        X_protein_val = X_protein_train_scaled[val_idx]
        X_mutation_val = X_mutation_train_scaled[val_idx]
        y_val = y_train[val_idx]
        
        train_val_idx = inner_split[0]
        X_protein_train_final = X_protein_train_scaled[train_val_idx]
        X_mutation_train_final = X_mutation_train_scaled[train_val_idx]
        y_train_final = y_train[train_val_idx]
        
        # Initialize best model and best score
        best_score = -np.inf
        best_model = None
        best_history = None
        best_params = {}
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc' if num_classes == 2 else 'val_accuracy',
            patience=patience,
            restore_best_weights=True,
            mode='max'
        )
        
        # Simple hyperparameter tuning
        for dropout in param_grid['dropout_rate']:
            for lr in param_grid['learning_rate']:
                for l2_reg in param_grid['l2_reg']:
                    if model_type == 'MOE':
                        for num_experts in param_grid['num_experts']:
                            # Create Mixture of Experts model
                            model = create_tf_moe_model(
                                protein_input_shape=(X_protein_train_final.shape[1],),
                                mutation_input_shape=(X_mutation_train_final.shape[1],),
                                num_classes=num_classes,
                                num_experts=num_experts,
                                dropout_rate=dropout,
                                l2_reg=l2_reg,
                                learning_rate=lr
                            )
                            
                            # Train model
                            history = model.fit(
                                [X_protein_train_final, X_mutation_train_final],
                                y_train_final,
                                validation_data=([X_protein_val, X_mutation_val], y_val),
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[early_stopping],
                                verbose=0
                            )
                            
                            # Evaluate on validation set
                            val_score = history.history['val_auc'][-1] if num_classes == 2 else history.history['val_accuracy'][-1]
                            
                            # Check if this is the best model so far
                            if val_score > best_score:
                                best_score = val_score
                                best_model = model
                                best_history = history
                                best_params = {
                                    'dropout_rate': dropout,
                                    'learning_rate': lr,
                                    'l2_reg': l2_reg,
                                    'num_experts': num_experts
                                }
                    else:  # Late Fusion model
                        # Create Late Fusion model
                        model = create_tf_late_fusion_model(
                            protein_input_shape=(X_protein_train_final.shape[1],),
                            mutation_input_shape=(X_mutation_train_final.shape[1],),
                            num_classes=num_classes,
                            dropout_rate=dropout,
                            l2_reg=l2_reg,
                            learning_rate=lr
                        )
                        
                        # Train model
                        history = model.fit(
                            [X_protein_train_final, X_mutation_train_final],
                            y_train_final,
                            validation_data=([X_protein_val, X_mutation_val], y_val),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        
                        # Evaluate on validation set
                        val_score = history.history['val_auc'][-1] if num_classes == 2 else history.history['val_accuracy'][-1]
                        
                        # Check if this is the best model so far
                        if val_score > best_score:
                            best_score = val_score
                            best_model = model
                            best_history = history
                            best_params = {
                                'dropout_rate': dropout,
                                'learning_rate': lr,
                                'l2_reg': l2_reg
                            }
        
        # Make predictions on test set using best model
        y_pred_proba = best_model.predict([X_protein_test_scaled, X_mutation_test_scaled], verbose=0)
        
        if num_classes == 2:
            # Binary classification
            y_pred_proba = y_pred_proba.flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Store predictions and probabilities
            all_predictions[test_idx] = y_pred
            all_probabilities[test_idx] = y_pred_proba
            
            # Calculate performance metrics
            fold_acc = accuracy_score(y_test, y_pred)
            fold_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Calculate ROC curve data for this fold
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            
            fold_results.append({
                'fold': fold,
                'accuracy': fold_acc,
                'auc': fold_auc,
                'best_params': best_params,
                'test_indices': test_idx
            })
        else:
            # Multi-class classification
            y_pred_classes = np.argmax(y_pred_proba, axis=1)
            all_predictions[test_idx] = y_pred_classes
            
            # Calculate accuracy for this fold
            fold_acc = accuracy_score(y_test, y_pred_classes)
            
            fold_results.append({
                'fold': fold,
                'accuracy': fold_acc,
                'best_params': best_params,
                'test_indices': test_idx
            })
    
    # Calculate overall metrics
    if num_classes == 2:
        # Binary classification
        overall_acc = accuracy_score(y, all_predictions)
        overall_auc = roc_auc_score(y, all_probabilities)
        
        # Calculate mean ROC curve for plotting
        from sklearn.metrics import roc_curve
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        
        for i in range(len(fpr_list)):
            mean_tpr += np.interp(mean_fpr, fpr_list[i], tpr_list[i])
        
        mean_tpr /= len(fpr_list)
        
        print(f"Completed {model_type} cancer detection model with overall AUC: {overall_auc:.4f}")
        
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
    else:
        # Multi-class classification
        overall_acc = accuracy_score(y, all_predictions)
        print(f"Completed {model_type} tissue localization model with overall accuracy: {overall_acc:.4f}")
        
        return {
            'fold_results': fold_results,
            'overall_accuracy': overall_acc,
            'predictions': all_predictions
        }

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
    X_protein = X[protein_features]
    X_omega = X[[omega_score_col]] if omega_score_col in X.columns else pd.DataFrame(index=X.index)
    
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
        
        try:
            # Extract protein data for this fold
            X_protein_train = X_protein.iloc[train_idx].copy()
            X_protein_test = X_protein.iloc[test_idx].copy()
            
            # Extract omega score data
            X_omega_train = X_omega.iloc[train_idx].copy()
            X_omega_test = X_omega.iloc[test_idx].copy()
            
            # Create indices for normalization
            train_indices = list(range(len(X_protein_train)))
            
            # Normalize protein data
            try:
                X_protein_train_norm, percentile_values = normalize_protein_data(
                    X_protein_train.reset_index(drop=True), clinical_df, training_indices=train_indices)
            except Exception as e:
                print(f"Error normalizing protein data: {str(e)}. Using original data.")
                X_protein_train_norm = X_protein_train.copy()
                percentile_values = {}
            
            # Apply same normalization to test data using parameters from training
            try:
                X_protein_test_norm, _ = normalize_protein_data(
                    X_protein_test.reset_index(drop=True), clinical_df, training_indices=None, percentile_values=percentile_values)
            except Exception as e:
                print(f"Error normalizing test protein data: {str(e)}. Using original data.")
                X_protein_test_norm = X_protein_test.copy()
            
            # Normalize mutation data if available
            try:
                # Extract mutation data
                X_mutation_train = X_omega_train.copy()
                X_mutation_test = X_omega_test.copy()
                
                # Normalize mutation data using numerical indices
                X_mutation_train_norm, detection_limits = normalize_mutation_data(
                    X_mutation_train.reset_index(drop=True), clinical_df, training_indices=train_indices)
                
                # Apply same normalization to test data using parameters from training
                X_mutation_test_norm, _ = normalize_mutation_data(
                    X_mutation_test.reset_index(drop=True), clinical_df, training_indices=None, stored_limits=detection_limits)
            except Exception as e:
                print(f"Error normalizing mutation data: {str(e)}. Using original data.")
                if omega_score_col in X.columns:
                    X_mutation_train_norm = X_mutation_train.copy()
                    X_mutation_test_norm = X_mutation_test.copy()
                else:
                    # Create empty DataFrames if omega_score is not available
                    X_mutation_train_norm = pd.DataFrame(index=X_protein_train.index)
                    X_mutation_test_norm = pd.DataFrame(index=X_protein_test.index)
                    # Add omega_score column with zeros
                    X_mutation_train_norm[omega_score_col] = 0
                    X_mutation_test_norm[omega_score_col] = 0
            
            # Combine protein and mutation features
            X_train_combined = pd.concat([X_protein_train_norm, X_mutation_train_norm], axis=1)
            X_test_combined = pd.concat([X_protein_test_norm, X_mutation_test_norm], axis=1)
            
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
        except Exception as e:
            print(f"Error in fold {fold}: {str(e)}. Using original data.")
            all_predictions[test_idx] = y_test
            all_probabilities[test_idx] = 0.5
            fold_results.append({
                'fold': fold,
                'accuracy': np.nan,
                'auc': np.nan,
                'best_params': {},
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
    Perform nested cross-validation for tissue localization using Random Forest or XGBoost.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame containing features for model
    y : pandas.Series or numpy.ndarray
        Target variable (cancer types)
    protein_features : list, optional
        List of protein features to use in the model
        If None, all features except explicitly excluded ones will be used
    omega_score_col : str, optional
        Name of the omega score column (default: 'omega_score')
    include_gender : bool, optional
        Whether to include gender as a feature (default: True)
    clinical_df : pandas.DataFrame, optional
        DataFrame with clinical information for stratified sampling
    outer_splits : int, optional
        Number of folds for outer cross-validation (default: 10)
    inner_splits : int, optional
        Number of folds for inner cross-validation (default: 6)
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
    # Check if protein_features is None, use all columns except excluded ones
    if protein_features is None:
        non_protein_cols = ['Patient_ID', 'Sample_ID', 'Tumor_type', 'AJCC_Stage', 
                            'CancerSEEK_Logistic_Regression_Score', 'CancerSEEK_Test_Result',
                            'omega_score', 'is_cancer']
        candidate_protein_features = [col for col in X.columns if col not in non_protein_cols]
        
        # Filter out boolean and non-numeric columns
        protein_features = []
        for col in candidate_protein_features:
            if (pd.api.types.is_numeric_dtype(X[col]) and 
                not pd.api.types.is_bool_dtype(X[col])):
                protein_features.append(col)
            else:
                print(f"Warning: Column '{col}' is not numeric or is boolean. Excluding from RF features.")
    
    # Include gender feature if requested
    gender_col = None
    if include_gender and 'Sex' in X.columns:
        gender_col = 'Sex'
        # Check if gender is already numeric
        if not pd.api.types.is_numeric_dtype(X[gender_col]):
            print(f"Warning: Sex column is not numeric. Will convert to numeric.")
    
    features = protein_features.copy()
    if omega_score_col in X.columns:
        # Check if omega_score is numeric before including
        if (pd.api.types.is_numeric_dtype(X[omega_score_col]) and 
            not pd.api.types.is_bool_dtype(X[omega_score_col])):
            features.append(omega_score_col)
        else:
            print(f"Warning: '{omega_score_col}' is not numeric or is boolean. Excluding from RF features.")
    
    print(f"Using {len(features)} features for {model_type} model")
    
    # If using XGBoost, set up parameters and encode labels
    if model_type == 'XGB':
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 200],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        base_model = XGBClassifier(random_state=random_state, eval_metric='mlogloss')
        
        # For XGBoost, perform label encoding since it requires numeric labels
        print("Encoding cancer type labels for XGBoost...")
        unique_classes = np.unique(y)
        class_to_index = {cls: i for i, cls in enumerate(unique_classes)}
        index_to_class = {i: cls for cls, i in class_to_index.items()}
        
        # Convert string labels to numeric
        if isinstance(y, pd.Series):
            y_for_training = np.array([class_to_index[label] for label in y.values])
        else:
            y_for_training = np.array([class_to_index[label] for label in y])
        
        classes = unique_classes
    else:  # Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        base_model = RandomForestClassifier(random_state=random_state)
        
        # Random Forest can handle string labels directly
        y_for_training = y
        classes = np.unique(y)
    
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
    # ADDED: Container for probabilities (num_samples x num_classes)
    all_probabilities = np.zeros((len(y), len(classes)))
    
    # Confusion matrices for each fold
    confusion_matrices = []
    feature_importances = []
    
    # Prepare splits for tqdm
    splits = list(outer_cv.split(X.values, y_for_training))  # Use y_for_training for stratification
    
    # Define feature extractors
    X_protein = X[protein_features]
    X_omega = X[[omega_score_col]] if omega_score_col in X.columns else pd.DataFrame(index=X.index)
    
    # Perform nested cross-validation
    for fold, (train_idx, test_idx) in enumerate(tqdm(splits, desc=f"{model_type} tissue localization CV folds")):
        # Access the original data for proper normalization
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
        
        # Extract features for this fold
        X_protein_train = X_protein.iloc[train_idx].copy()
        X_protein_test = X_protein.iloc[test_idx].copy()
        
        # Create indices for normalization
        train_indices = list(range(len(X_protein_train)))
        
        # Normalize protein data - prevents data leakage by using only training data for normalization parameters
        X_protein_train_norm, percentile_values = normalize_protein_data(
            X_protein_train.reset_index(drop=True), clinical_df, training_indices=train_indices)
        
        # Apply same normalization to test data using parameters from training
        X_protein_test_norm, _ = normalize_protein_data(
            X_protein_test.reset_index(drop=True), clinical_df, training_indices=None, percentile_values=percentile_values)
        
        # Normalize mutation data if available
        X_mutation_train = X_omega.iloc[train_idx].copy()
        X_mutation_test = X_omega.iloc[test_idx].copy()
        
        # Normalize mutation data using numerical indices
        X_mutation_train_norm, detection_limits = normalize_mutation_data(
            X_mutation_train.reset_index(drop=True), clinical_df, training_indices=train_indices)
        
        # Apply same normalization to test data using parameters from training
        X_mutation_test_norm, _ = normalize_mutation_data(
            X_mutation_test.reset_index(drop=True), clinical_df, training_indices=None, stored_limits=detection_limits)
        
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
        
        # Combine protein and omega features for the traditional models
        X_train_combined = np.hstack([X_protein_train_scaled, X_omega_train_scaled])
        X_test_combined = np.hstack([X_protein_test_scaled, X_omega_test_scaled])
        
        # Setup grid search with inner CV
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        # Find best hyperparameters
        grid_search.fit(X_train_combined, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions on test set
        y_pred = best_model.predict(X_test_combined)
        # ADDED: Get predicted probabilities
        y_pred_proba = best_model.predict_proba(X_test_combined)
        
        # For XGBoost, convert predicted class indices back to original labels
        if model_type == 'XGB':
            y_pred = np.array([index_to_class[idx] for idx in y_pred])
        
        # Store predictions
        for i, idx in enumerate(test_idx):
            all_predictions[idx] = y_pred[i]
            # ADDED: Store probabilities for the corresponding sample
            # Ensure the order of probabilities matches the order of classes
            if hasattr(best_model, 'classes_'):
                class_order = best_model.classes_
            else:
                class_order = classes # Assume the order matches the unique classes found earlier
            
            # Map probabilities to the correct class index in all_probabilities
            prob_dict = dict(zip(class_order, y_pred_proba[i]))
            for class_idx, class_name in enumerate(classes):
                all_probabilities[idx, class_idx] = prob_dict.get(class_name, 0.0)
        
        # Calculate confusion matrix for this fold
        fold_cm = confusion_matrix(y_test_original, y_pred, labels=classes)
        confusion_matrices.append({
            'fold': fold,
            'confusion_matrix': fold_cm
        })
        
        # Calculate accuracy for this fold
        fold_acc = accuracy_score(y_test_original, y_pred)
        fold_results.append({
            'fold': fold,
            'accuracy': fold_acc,
            'test_indices': test_idx
        })
        
        # Calculate feature importances (only for RF or XGB)
        if hasattr(best_model, 'feature_importances_'):
            feature_importances.append({
                'fold': fold,
                'feature_names': features,
                'importances': best_model.feature_importances_
            })
    
    # Calculate overall metrics
    # Make sure all predictions are a valid type (string/object)
    mask = all_predictions != ''
    if np.any(mask):
        # Calculate accuracy only for non-empty predictions
        overall_acc = accuracy_score(y[mask], all_predictions[mask])
        
        # Calculate overall confusion matrix
        overall_cm = confusion_matrix(y[mask], all_predictions[mask], labels=classes)
    else:
        overall_acc = np.nan
        overall_cm = np.zeros((len(classes), len(classes)))
    
    # Get feature names
    feature_names = features
    
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
        # ADDED: Return probabilities
        'probabilities': all_probabilities,
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
                                              outer_splits=10, inner_splits=4, random_state=42,
                                              standardize_features=True, log_transform=None,
                                              detection_model='LR', localization_model='RF',
                                              detection_features=None):
    """
    Perform combined cancer detection and tissue localization with nested cross-validation.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        DataFrame containing features for model
    y_cancer_status : pandas.Series or numpy.ndarray
        Binary target variable (cancer vs normal)
    y_cancer_type : pandas.Series or numpy.ndarray
        Categorical target variable for cancer types
    clinical_df : pandas.DataFrame, optional
        DataFrame with clinical information for stratified sampling
    outer_splits : int, optional
        Number of folds for outer cross-validation (default: 10)
    inner_splits : int, optional
        Number of folds for inner cross-validation (default: 4)
    random_state : int, optional
        Random seed for reproducibility (default: 42)
    standardize_features : bool, optional
        Whether to standardize features (default: True)
    log_transform : str, optional
        Type of log transformation to apply to protein features ('log2', 'log10', or None)
    detection_model : str, optional
        Type of model to use for cancer detection ('LR', 'XGB', 'TF', 'MOE') (default: 'LR')
    localization_model : str, optional
        Type of model to use for tissue localization ('RF', 'XGB', 'TF', 'MOE') (default: 'RF')
    detection_features : list, optional
        List of features to use for cancer detection model
        
    Returns:
    --------
    dict
        Dictionary containing combined results from both models
    """
    print(f"\nRunning cancer detection model ({detection_model})...")
    
    # Step 1: Run cancer detection model
    if detection_model == 'LR':
        detection_results = nested_cross_validation(
            X, y_cancer_status, 
            protein_features=detection_features,
            omega_score_col='omega_score' if 'omega_score' in X.columns else None,
            clinical_df=clinical_df,
            outer_splits=outer_splits, 
            inner_splits=inner_splits,
            random_state=random_state,
            standardize_features=standardize_features,
            log_transform=log_transform,
            model_type='LR'
        )
    elif detection_model == 'XGB':
        detection_results = nested_cross_validation(
            X, y_cancer_status, 
            protein_features=detection_features,
            omega_score_col='omega_score' if 'omega_score' in X.columns else None,
            clinical_df=clinical_df, 
            outer_splits=outer_splits, 
            inner_splits=inner_splits,
            random_state=random_state,
            standardize_features=standardize_features,
            log_transform=log_transform,
            model_type='XGB'
        )
    elif detection_model in ['TF', 'MOE']:
        # If TensorFlow models are requested but not available, fall back to LR
        if not TENSORFLOW_AVAILABLE:
            print(f"Warning: {detection_model} requested but TensorFlow not available. Falling back to LR.")
            detection_model = 'LR'
            detection_results = nested_cross_validation(
                X, y_cancer_status, 
                protein_features=detection_features,
                omega_score_col='omega_score' if 'omega_score' in X.columns else None,
                clinical_df=clinical_df, 
                outer_splits=outer_splits, 
                inner_splits=inner_splits,
                random_state=random_state,
                standardize_features=standardize_features,
                log_transform=log_transform,
                model_type='LR'
            )
        else:
            # Use the TensorFlow model implementation
            detection_results = tf_cross_validation(
                X, y_cancer_status, 
                protein_features=detection_features,
                omega_score_col='omega_score' if 'omega_score' in X.columns else None,
                clinical_df=clinical_df, 
                outer_splits=outer_splits, 
                inner_splits=inner_splits,
                random_state=random_state,
                standardize_features=standardize_features,
                log_transform=log_transform,
                model_type=detection_model,  # 'TF' or 'MOE'
                num_classes=2  # Binary classification for detection
            )
    else:
        raise ValueError(f"Unsupported detection model type: {detection_model}")
    
    print(f"\nRunning tissue localization model ({localization_model})...")
    
    # Step 2: Run tissue localization model
    # Filter to only include cancer samples for localization
    cancer_mask = y_cancer_status == 1
    X_cancer = X[cancer_mask].copy()
    y_cancer_type_filtered = y_cancer_type[cancer_mask].copy()
    
    # Make sure we have clinical data for cancer samples only
    clinical_df_cancer = None
    if clinical_df is not None:
        clinical_df_cancer = clinical_df[clinical_df['Tumor_type'] != 'Normal'].copy()
    
    if localization_model == 'RF':
        localization_results = nested_cross_validation_rf(
            X_cancer, 
            y_cancer_type_filtered,
            protein_features=detection_features,  # Can use same or different features
            omega_score_col='omega_score' if 'omega_score' in X.columns else None,
            include_gender=True,
            clinical_df=clinical_df_cancer,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            random_state=random_state,
            standardize_features=standardize_features,
            log_transform=log_transform,
            model_type='RF'
        )
    elif localization_model == 'XGB':
        localization_results = nested_cross_validation_rf(
            X_cancer, 
            y_cancer_type_filtered,
            protein_features=detection_features,  # Can use same or different features
            omega_score_col='omega_score' if 'omega_score' in X.columns else None,
            include_gender=True,
            clinical_df=clinical_df_cancer,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            random_state=random_state,
            standardize_features=standardize_features,
            log_transform=log_transform,
            model_type='XGB'
        )
    elif localization_model in ['TF', 'MOE']:
        # If TensorFlow models are requested but not available, fall back to RF
        if not TENSORFLOW_AVAILABLE:
            print(f"Warning: {localization_model} requested but TensorFlow not available. Falling back to RF.")
            localization_model = 'RF'
            localization_results = nested_cross_validation_rf(
                X_cancer, 
                y_cancer_type_filtered,
                protein_features=detection_features,
                omega_score_col='omega_score' if 'omega_score' in X.columns else None,
                include_gender=True,
                clinical_df=clinical_df_cancer,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                random_state=random_state,
                standardize_features=standardize_features,
                log_transform=log_transform,
                model_type='RF'
            )
        else:
            # Get unique cancer types for determining num_classes
            unique_cancer_types = np.unique(y_cancer_type_filtered)
            num_classes = len(unique_cancer_types)
            
            # Create label encoder for cancer types
            type_to_idx = {cancer_type: i for i, cancer_type in enumerate(unique_cancer_types)}
            y_cancer_type_numeric = np.array([type_to_idx[t] for t in y_cancer_type_filtered])
            
            # Use the TensorFlow model implementation
            localization_results = tf_cross_validation(
                X_cancer, 
                y_cancer_type_numeric,
                protein_features=detection_features,
                omega_score_col='omega_score' if 'omega_score' in X.columns else None,
                clinical_df=clinical_df_cancer,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                random_state=random_state,
                standardize_features=standardize_features,
                log_transform=log_transform,
                model_type=localization_model,  # 'TF' or 'MOE'
                num_classes=num_classes  # Multi-class for localization
            )
            
            # Convert numeric predictions back to class labels
            idx_to_type = {i: cancer_type for cancer_type, i in type_to_idx.items()}
            localization_results['original_predictions'] = localization_results['predictions'].copy()
            localization_results['predictions'] = np.array([idx_to_type[p] for p in localization_results['predictions']])
    else:
        raise ValueError(f"Unsupported localization model type: {localization_model}")
    
    # Step 3: Combine results
    combined_results = {
        'detection_model': detection_model,
        'localization_model': localization_model,
        'detection_results': detection_results,
        'localization_results': localization_results
    }
    
    return combined_results

