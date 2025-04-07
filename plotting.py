import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, accuracy_score
from scipy.stats import norm
import statsmodels.stats.proportion as smp
from matplotlib.patches import Patch
import pickle

def plot_roc_curve(y_true, y_score, title="ROC Curve", figsize=(10, 8), model_type=None, save_data=None, target_specificity=0.99, standardized=None):
    """
    Plot ROC curve for a binary classifier.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Target scores (probability estimates of the positive class)
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size
    model_type : str, optional
        Type of model used for detection (e.g., 'LR', 'XGB')
    save_data : str, optional
        Path to save plotting data (CSV if ends with '.csv', pickle otherwise)
    target_specificity : float, optional
        Target specificity for displaying marker and CI (default: 0.99)
    standardized : bool, optional
        Indicates if features were standardized (default: None)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Round target specificity to 3 significant figures for display
    specificity_rounded = round(target_specificity * 1000) / 1000
    
    # Calculate sensitivity at target specificity
    sens_results = calculate_sensitivity_at_specificity(y_true, y_score, target_specificity=target_specificity)
    
    # Find the point on the curve closest to the target specificity
    target_fpr = 1 - target_specificity
    idx = np.argmin(np.abs(fpr - target_fpr))
    spec_point_x = fpr[idx]
    spec_point_y = tpr[idx]
    
    # Add red dot at target specificity point
    ax.plot(spec_point_x, spec_point_y, 'ro', markersize=8)
    
    # Add vertical error bars for confidence intervals
    ax.errorbar(spec_point_x, spec_point_y, 
                yerr=[[spec_point_y - sens_results['ci_low']], 
                      [sens_results['ci_high'] - spec_point_y]],
                fmt='none', color='red', capsize=5, capthick=2, elinewidth=2)
    
    # Add labels and legend
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    
    # Update title with model type and standardization status if provided
    final_title = title
    if model_type:
        if "ROC Curve" in final_title and "(" not in final_title:
            final_title = f"ROC Curve for CancerSEEK ({model_type})"
    if standardized is not None:
        std_text = " (Standardized)" if standardized else " (Not Standardized)"
        final_title += std_text
    
    ax.set_title(final_title)
    
    # Get the specificity value for display (rounded to 3 sig figs)
    spec_percent = int(specificity_rounded * 1000)
    
    # Update legend to include sensitivity at target specificity with CIs
    sens_label = f'Sensitivity @ {spec_percent/10:.1f}% Specificity = {sens_results["sensitivity"]:.3f} (95% CI: {sens_results["ci_low"]:.3f}-{sens_results["ci_high"]:.3f})'
    ax.legend(['ROC curve (AUC = {:.3f})'.format(roc_auc), 'Random Chance', sens_label], loc="lower right")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # If save_data is provided, update with specificity in filename
    if save_data:
        # Add specificity and standardization status to filename
        spec_suffix = f"_{spec_percent}"
        std_suffix = "_std" if standardized else "_no_std" if standardized is not None else ""
        
        if save_data.endswith('.csv'):
            base_name = save_data[:-4]
            save_data = f"{base_name}{spec_suffix}{std_suffix}.csv"
        elif save_data.endswith('.pkl'):
            base_name = save_data[:-4]
            save_data = f"{base_name}{spec_suffix}{std_suffix}.pkl"
        else:
            save_data = f"{save_data}{spec_suffix}{std_suffix}.pkl"
        
        plot_data = {
            'fpr': fpr,
            'tpr': tpr, 
            'thresholds': thresholds,
            'auc': roc_auc,
            'sensitivity_at_target_spec': sens_results,
            'specificity_rounded': specificity_rounded,
            'standardized': standardized
        }
        
        if save_data.endswith('.csv'):
            # Save as CSV
            pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'threshold': thresholds if len(thresholds) == len(fpr) else np.append(thresholds, np.nan)
            }).to_csv(save_data, index=False)
            
            # Save metadata separately since it doesn't fit well in CSV format
            meta_data = {
                'auc': roc_auc,
                'sensitivity_at_target_spec': sens_results,
                'model_type': model_type,
                'specificity_rounded': specificity_rounded,
                'standardized': standardized
            }
            meta_save_path = save_data.replace('.csv', '_meta.pkl')
            with open(meta_save_path, 'wb') as f:
                pickle.dump(meta_data, f)
        else:
            # Save as pickle
            save_path = save_data if save_data.endswith('.pkl') else save_data + '.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(plot_data, f)
    
    return fig, ax, spec_percent

def calculate_sensitivity_at_specificity(y_true, y_score, target_specificity=0.99):
    """
    Calculate sensitivity at a target specificity.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Target scores (probability estimates of the positive class)
    target_specificity : float, optional
        Target specificity value (default: 0.99)
        
    Returns:
    --------
    dict
        Dictionary containing sensitivity, threshold, and confidence intervals
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Round the target specificity to 3 significant figures
    target_specificity_rounded = round(target_specificity * 1000) / 1000
    
    # Find the threshold that gives the target specificity
    target_fpr = 1 - target_specificity
    # Find the index where FPR is closest to our target
    idx = np.argmin(np.abs(fpr - target_fpr))
    
    # Get the corresponding sensitivity and threshold
    sensitivity = tpr[idx]
    threshold = thresholds[idx]
    
    # Calculate exact specificity achieved
    specificity_achieved = 1 - fpr[idx]
    
    # Count true positives and total positives for confidence interval
    # Apply the threshold to the scores
    y_pred = (y_score >= threshold).astype(int)
    
    # Count true positives (cancer patients classified as cancer)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # Total cancer patients
    p = np.sum(y_true == 1)
    
    # Calculate Clopper-Pearson confidence interval
    ci_low, ci_high = smp.proportion_confint(tp, p, alpha=0.05, method='beta')
    
    # Count true negatives and total negatives for specificity confidence interval
    tn = np.sum((y_true == 0) & (y_pred == 0))
    n = np.sum(y_true == 0)
    spec_ci_low, spec_ci_high = smp.proportion_confint(tn, n, alpha=0.05, method='beta')
    
    return {
        'sensitivity': sensitivity,
        'threshold': threshold,
        'specificity': specificity_achieved,
        'specificity_rounded': target_specificity_rounded,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'spec_ci_low': spec_ci_low,
        'spec_ci_high': spec_ci_high,
        'true_positives': tp,
        'total_positives': p,
        'true_negatives': tn,
        'total_negatives': n
    }

def calculate_sensitivity_by_subtype(y_true, y_score, subtypes, target_specificity=0.99):
    """
    Calculate sensitivity at a target specificity for each cancer subtype.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (1 for cancer, 0 for normal)
    y_score : array-like
        Target scores (probability estimates of the positive class)
    subtypes : array-like
        Cancer subtypes for each sample (empty string or NaN for normal samples)
    target_specificity : float, optional
        Target specificity value (default: 0.99)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with sensitivity and confidence intervals for each subtype
    """
    # Find threshold for target specificity on the entire dataset
    result_all = calculate_sensitivity_at_specificity(y_true, y_score, target_specificity)
    threshold = result_all['threshold']
    
    # Initialize results list
    results = []
    
    # Add overall result
    results.append({
        'Cancer_Type': 'All Cancers',
        'Sensitivity': result_all['sensitivity'],
        'CI_Low': result_all['ci_low'],
        'CI_High': result_all['ci_high'],
        'True_Positives': result_all['true_positives'],
        'Total_Positives': result_all['total_positives'],
        'Specificity': result_all['specificity'],
        'Specificity_Rounded': result_all['specificity_rounded'],
        'Spec_CI_Low': result_all['spec_ci_low'],
        'Spec_CI_High': result_all['spec_ci_high'],
        'True_Negatives': result_all['true_negatives'],
        'Total_Negatives': result_all['total_negatives']
    })
    
    # Get unique cancer types
    unique_subtypes = np.unique([s for s in subtypes if s and not pd.isna(s)])
    
    # For each subtype, calculate sensitivity
    for subtype in unique_subtypes:
        # Filter to only this subtype and normal samples
        subtype_mask = (subtypes == subtype) | (y_true == 0)
        
        if np.sum(subtypes == subtype) == 0:
            continue
        
        # Apply threshold from the overall dataset
        y_pred_subtype = (y_score[subtype_mask] >= threshold).astype(int)
        y_true_subtype = y_true[subtype_mask]
        
        # Find samples of this subtype
        cancer_mask = (subtypes == subtype)
        
        # Calculate TP and P for this subtype
        tp = np.sum((cancer_mask) & (y_score >= threshold))
        p = np.sum(cancer_mask)
        
        # Calculate sensitivity for this subtype
        if p > 0:
            sensitivity = tp / p
            # Calculate Clopper-Pearson confidence interval
            ci_low, ci_high = smp.proportion_confint(tp, p, alpha=0.05, method='beta')
        else:
            sensitivity = np.nan
            ci_low, ci_high = np.nan, np.nan
        
        results.append({
            'Cancer_Type': subtype,
            'Sensitivity': sensitivity,
            'CI_Low': ci_low,
            'CI_High': ci_high,
            'True_Positives': tp,
            'Total_Positives': p,
            'Specificity': result_all['specificity'],
            'Specificity_Rounded': result_all['specificity_rounded'],
            'Spec_CI_Low': result_all['spec_ci_low'],
            'Spec_CI_High': result_all['spec_ci_high'],
            'True_Negatives': result_all['true_negatives'],
            'Total_Negatives': result_all['total_negatives']
        })
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def plot_sensitivity_by_subtype(sensitivity_df, figsize=(12, 8), model_type=None, standardized=None):
    """
    Plot sensitivity by cancer subtype with confidence intervals.
    
    Parameters:
    -----------
    sensitivity_df : pd.DataFrame
        DataFrame with sensitivity data from calculate_sensitivity_by_subtype
    figsize : tuple, optional
        Figure size
    model_type : str, optional
        Type of model used for detection (e.g., 'LR', 'XGB')
    standardized : bool, optional
        Indicates if features were standardized (default: None)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out 'All Cancers' and sort by sensitivity (descending)
    df_filtered = sensitivity_df[sensitivity_df['Cancer_Type'] != 'All Cancers']
    df_sorted = df_filtered.sort_values('Sensitivity', ascending=False)
    
    # Get the specificity value for display (rounded to 3 sig figs)
    spec_rounded = sensitivity_df['Specificity_Rounded'].iloc[0]
    spec_percent = int(spec_rounded * 1000)
    
    # Create a colormap for consistent coloring
    colors = plt.cm.tab20.colors[:len(df_sorted)]
    
    # Plot bar chart with colormap
    bars = ax.bar(df_sorted['Cancer_Type'], df_sorted['Sensitivity'], color=colors)
    
    # Add error bars
    ax.errorbar(
        df_sorted['Cancer_Type'], 
        df_sorted['Sensitivity'],
        yerr=[df_sorted['Sensitivity'] - df_sorted['CI_Low'], 
              df_sorted['CI_High'] - df_sorted['Sensitivity']],
        fmt='none', 
        color='black', 
        capsize=5
    )
    
    # Add sensitivity values as text on each bar
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(i, row['Sensitivity'] / 2, f"{row['Sensitivity']:.2f}", 
                ha='center', va='center', color='white', fontweight='bold')
        
        # Also add the total count as smaller text just below the sensitivity value
        ax.text(i, row['Sensitivity'] / 2 - 0.05, f"({row['True_Positives']}/{row['Total_Positives']})", 
                ha='center', va='center', color='white', fontsize=8)
    
    # Add labels and title
    ax.set_xlabel('Cancer Type')
    ax.set_ylabel(f'Sensitivity at {spec_percent/10:.1f}% Specificity')
    
    # Update title with model type and standardization status if provided
    final_title = f"Sensitivity by Cancer Type at {spec_percent/10:.1f}% Specificity"
    if model_type:
        final_title = f"Sensitivity by Cancer Type at {spec_percent/10:.1f}% Specificity ({model_type})"
    if standardized is not None:
        std_text = " (Standardized)" if standardized else " (Not Standardized)"
        final_title += std_text
    ax.set_title(final_title)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Set y-axis limit
    ax.set_ylim([0, 1.05])
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax, spec_percent

def calculate_sensitivity_by_stage(y_true, y_score, stages, target_specificity=0.99):
    """
    Calculate sensitivity at a target specificity for each cancer stage.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (1 for cancer, 0 for normal)
    y_score : array-like
        Target scores (probability estimates of the positive class)
    stages : array-like
        Cancer stages for each sample (empty string or NaN for normal samples)
    target_specificity : float, optional
        Target specificity value (default: 0.99)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with sensitivity and confidence intervals for each stage
    """
    # Find threshold for target specificity on the entire dataset
    result_all = calculate_sensitivity_at_specificity(y_true, y_score, target_specificity)
    threshold = result_all['threshold']
    
    # Initialize results list
    results = []
    
    # Add overall result
    results.append({
        'Stage': 'All Stages',
        'Sensitivity': result_all['sensitivity'],
        'CI_Low': result_all['ci_low'],
        'CI_High': result_all['ci_high'],
        'True_Positives': result_all['true_positives'],
        'Total_Positives': result_all['total_positives'],
        'Specificity': result_all['specificity'],
        'Specificity_Rounded': result_all['specificity_rounded'],
        'Spec_CI_Low': result_all['spec_ci_low'],
        'Spec_CI_High': result_all['spec_ci_high'],
        'True_Negatives': result_all['true_negatives'],
        'Total_Negatives': result_all['total_negatives']
    })
    
    # Get unique cancer stages and ensure proper ordering
    # Fix: Filter out NaN values and convert to strings to avoid type comparison issues
    stages_array = np.array(stages)
    valid_stages = [s for s in stages_array if isinstance(s, str) or (not pd.isna(s) and s)]
    unique_stages = []
    for stage in ['I', 'II', 'III', 'IV']:
        if stage in valid_stages:
            unique_stages.append(stage)
    
    # For each stage, calculate sensitivity
    for stage in unique_stages:
        # Find samples of this stage
        stage_mask = (stages == stage)
        
        if np.sum(stage_mask) == 0:
            continue
        
        # Calculate TP and P for this stage
        tp = np.sum((stage_mask) & (y_score >= threshold))
        p = np.sum(stage_mask)
        
        # Calculate sensitivity for this stage
        if p > 0:
            sensitivity = tp / p
            # Calculate Clopper-Pearson confidence interval
            ci_low, ci_high = smp.proportion_confint(tp, p, alpha=0.05, method='beta')
        else:
            sensitivity = np.nan
            ci_low, ci_high = np.nan, np.nan
        
        results.append({
            'Stage': stage,
            'Sensitivity': sensitivity,
            'CI_Low': ci_low,
            'CI_High': ci_high,
            'True_Positives': tp,
            'Total_Positives': p,
            'Specificity': result_all['specificity'],
            'Specificity_Rounded': result_all['specificity_rounded'],
            'Spec_CI_Low': result_all['spec_ci_low'],
            'Spec_CI_High': result_all['spec_ci_high'],
            'True_Negatives': result_all['true_negatives'],
            'Total_Negatives': result_all['total_negatives']
        })
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def plot_sensitivity_by_stage(sensitivity_df, figsize=(10, 6), model_type=None, standardized=None):
    """
    Plot sensitivity by cancer stage with confidence intervals.
    
    Parameters:
    -----------
    sensitivity_df : pd.DataFrame
        DataFrame with sensitivity data from calculate_sensitivity_by_stage
    figsize : tuple, optional
        Figure size
    model_type : str, optional
        Type of model used for detection (e.g., 'LR', 'XGB')
    standardized : bool, optional
        Indicates if features were standardized (default: None)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter out 'All Stages'
    df_filtered = sensitivity_df[sensitivity_df['Stage'] != 'All Stages']
    
    # Get the specificity value for display (rounded to 3 sig figs)
    spec_rounded = sensitivity_df['Specificity_Rounded'].iloc[0]
    spec_percent = int(spec_rounded * 1000)
    
    # Extract stage order (ensure proper ordering)
    stage_order = []
    for stage in ['I', 'II', 'III', 'IV']:
        if stage in df_filtered['Stage'].tolist():
            stage_order.append(stage)
    
    # Filter and order the DataFrame
    df_sorted = df_filtered[df_filtered['Stage'].isin(stage_order)].set_index('Stage').loc[stage_order].reset_index()
    
    # Define consistent colors for stages
    stage_colors = {
        'I': 'lightgreen',
        'II': 'yellowgreen',
        'III': 'orange',
        'IV': 'tomato'
    }
    
    # Get colors for the bars
    bar_colors = [stage_colors[stage] for stage in df_sorted['Stage']]
    
    # Plot bar chart
    bars = ax.bar(df_sorted['Stage'], df_sorted['Sensitivity'], color=bar_colors)
    
    # Add error bars
    ax.errorbar(
        df_sorted['Stage'], 
        df_sorted['Sensitivity'],
        yerr=[df_sorted['Sensitivity'] - df_sorted['CI_Low'], 
              df_sorted['CI_High'] - df_sorted['Sensitivity']],
        fmt='none', 
        color='black', 
        capsize=5
    )
    
    # Add sensitivity values as text on each bar
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(i, row['Sensitivity'] / 2, f"{row['Sensitivity']:.2f}", 
                ha='center', va='center', color='black', fontweight='bold')
        
        # Also add the total count as smaller text just below the sensitivity value
        ax.text(i, row['Sensitivity'] / 2 - 0.05, f"({row['True_Positives']}/{row['Total_Positives']})", 
                ha='center', va='center', color='black', fontsize=8)
    
    # Add labels and title
    ax.set_xlabel('Cancer Stage')
    ax.set_ylabel(f'Sensitivity at {spec_percent/10:.1f}% Specificity')
    
    # Update title with model type and standardization status if provided
    final_title = f"Sensitivity by Cancer Stage at {spec_percent/10:.1f}% Specificity"
    if model_type:
        final_title = f"Sensitivity by Cancer Stage at {spec_percent/10:.1f}% Specificity ({model_type})"
    if standardized is not None:
        std_text = " (Standardized)" if standardized else " (Not Standardized)"
        final_title += std_text
    ax.set_title(final_title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Set y-axis limit
    ax.set_ylim([0, 1.05])
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax, spec_percent

def create_performance_summary_csv(sensitivity_by_stage_df, output_path, model_type=None, standardized=None):
    """
    Create a CSV summary of the performance metrics with proper formatting.
    
    Parameters:
    -----------
    sensitivity_by_stage_df : pd.DataFrame
        DataFrame with sensitivity data from calculate_sensitivity_by_stage
    output_path : str
        Path to save the CSV file
    model_type : str, optional
        Type of model used for detection (e.g., 'LR', 'XGB')
    standardized : bool, optional
        Indicates if features were standardized (default: None)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with formatted performance metrics
    """
    # Create a new DataFrame for the summary with proper row indexing
    summary_df = pd.DataFrame(columns=["Estimate %", "95% CI", "n/N"])
    
    # Extract specificity info
    spec_row = sensitivity_by_stage_df.iloc[0]
    spec = spec_row['Specificity']
    spec_rounded = spec_row['Specificity_Rounded']
    spec_ci_low = spec_row['Spec_CI_Low']
    spec_ci_high = spec_row['Spec_CI_High']
    spec_n = spec_row['True_Negatives']
    spec_N = spec_row['Total_Negatives']
    
    # Format the specificity row
    summary_df.loc["specificity", "Estimate %"] = f"{spec*100:.1f}%"
    summary_df.loc["specificity", "95% CI"] = f"{spec_ci_low*100:.1f}%, {spec_ci_high*100:.1f}%"
    summary_df.loc["specificity", "n/N"] = f"{spec_n:,} / {spec_N:,}"
    
    # Extract sensitivity for all cancers
    all_cancers = sensitivity_by_stage_df[sensitivity_by_stage_df['Stage'] == 'All Stages'].iloc[0]
    sens_all = all_cancers['Sensitivity']
    sens_all_ci_low = all_cancers['CI_Low']
    sens_all_ci_high = all_cancers['CI_High']
    sens_all_n = all_cancers['True_Positives']
    sens_all_N = all_cancers['Total_Positives']
    
    # Format the all cancers sensitivity row
    summary_df.loc["sensitivity-all", "Estimate %"] = f"{sens_all*100:.1f}%"
    summary_df.loc["sensitivity-all", "95% CI"] = f"{sens_all_ci_low*100:.1f}%, {sens_all_ci_high*100:.1f}%"
    summary_df.loc["sensitivity-all", "n/N"] = f"{sens_all_n:,} / {sens_all_N:,}"
    
    # Extract and format sensitivity by stage
    for stage, stage_name in [('I', 'sensitivity-stage1'), ('II', 'sensitivity-stage2'), 
                              ('III', 'sensitivity-stage3'), ('IV', 'sensitivity-stage4')]:
        stage_row = sensitivity_by_stage_df[sensitivity_by_stage_df['Stage'] == stage]
        if len(stage_row) > 0:
            stage_data = stage_row.iloc[0]
            sens = stage_data['Sensitivity']
            sens_ci_low = stage_data['CI_Low']
            sens_ci_high = stage_data['CI_High']
            sens_n = stage_data['True_Positives']
            sens_N = stage_data['Total_Positives']
            
            summary_df.loc[stage_name, "Estimate %"] = f"{sens*100:.1f}%"
            summary_df.loc[stage_name, "95% CI"] = f"{sens_ci_low*100:.1f}%, {sens_ci_high*100:.1f}%"
            summary_df.loc[stage_name, "n/N"] = f"{sens_n:,} / {sens_N:,}"
    
    # Handle Stage I, II combined if we have both stages
    if 'I' in sensitivity_by_stage_df['Stage'].values and 'II' in sensitivity_by_stage_df['Stage'].values:
        # Calculate combined stats for Stage I and II
        stage_i = sensitivity_by_stage_df[sensitivity_by_stage_df['Stage'] == 'I'].iloc[0]
        stage_ii = sensitivity_by_stage_df[sensitivity_by_stage_df['Stage'] == 'II'].iloc[0]
        
        stage_i_tp = stage_i['True_Positives']
        stage_i_total = stage_i['Total_Positives']
        stage_ii_tp = stage_ii['True_Positives']
        stage_ii_total = stage_ii['Total_Positives']
        
        combined_tp = stage_i_tp + stage_ii_tp
        combined_total = stage_i_total + stage_ii_total
        combined_sens = combined_tp / combined_total if combined_total > 0 else 0
        
        # Calculate CI for combined
        combined_ci_low, combined_ci_high = smp.proportion_confint(combined_tp, combined_total, alpha=0.05, method='beta')
        
        summary_df.loc["sensitivity-stage1-2", "Estimate %"] = f"{combined_sens*100:.1f}%"
        summary_df.loc["sensitivity-stage1-2", "95% CI"] = f"{combined_ci_low*100:.1f}%, {combined_ci_high*100:.1f}%"
        summary_df.loc["sensitivity-stage1-2", "n/N"] = f"{combined_tp:,} / {combined_total:,}"
    
    # Reset index and add row number index
    summary_df = summary_df.reset_index().rename(columns={"index": "Row"})
    
    # Add specificity to filename if provided
    spec_suffix = f"_{int(spec_rounded*1000)}"
    if model_type:
        base_name = output_path.rsplit('.', 1)[0]
        output_path = f"{base_name}_{model_type}{spec_suffix}.csv"
    else:
        base_name = output_path.rsplit('.', 1)[0]
        output_path = f"{base_name}{spec_suffix}.csv"
    
    # Save to CSV with properly quoted strings
    summary_df.to_csv(output_path, index=True, quoting=1)  # quoting=1 is QUOTE_ALL
    
    return summary_df

def plot_tissue_localization_accuracy(y_true, y_pred_proba, cancer_types=None, figsize=(12, 8),
                                     detection_model=None, localization_model=None, standardized=None):
    """
    Plot tissue localization accuracy for the top and top 2 predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True cancer types
    y_pred_proba : array-like or dict
        Predicted probabilities for each cancer type
        Can be a 2D array (samples x classes) or a dict mapping sample indices to probability dicts
    cancer_types : list, optional
        List of cancer type names. If None, inferred from unique values in y_true
    figsize : tuple, optional
        Figure size
    detection_model : str, optional
        Type of model used for detection (e.g., 'LR', 'XGB')
    localization_model : str, optional
        Type of model used for tissue localization (e.g., 'RF', 'XGB')
    standardized : bool, optional
        Indicates if features were standardized (default: None)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # If cancer_types not provided, infer from y_true
    if cancer_types is None:
        cancer_types = np.unique(y_true)
    
    # Initialize results dictionary
    results = {}
    
    # Process predictions and calculate accuracy
    if isinstance(y_pred_proba, dict):
        # Handle dict-of-dicts format
        for cancer_type in cancer_types:
            # Filter to samples of this cancer type
            type_indices = [i for i, t in enumerate(y_true) if t == cancer_type]
            
            if not type_indices:
                continue
                
            # Count correct top predictions
            top_correct = 0
            top2_correct = 0
            
            for idx in type_indices:
                if idx not in y_pred_proba:
                    continue
                    
                # Get probability dict for this sample
                probs = y_pred_proba[idx]
                
                # Convert to sorted list of (cancer_type, prob) tuples
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                
                # Check if top prediction is correct
                if sorted_probs[0][0] == cancer_type:
                    top_correct += 1
                    top2_correct += 1
                # Check if second prediction is correct
                elif len(sorted_probs) > 1 and sorted_probs[1][0] == cancer_type:
                    top2_correct += 1
            
            # Calculate accuracies
            top_acc = top_correct / len(type_indices) if type_indices else 0
            top2_acc = top2_correct / len(type_indices) if type_indices else 0
            
            results[cancer_type] = {
                'top': top_acc,
                'top2': top2_acc,
                'count': len(type_indices)
            }
    else:
        # Handle array format (samples x classes)
        for cancer_type in cancer_types:
            # Filter to samples of this cancer type
            type_mask = np.array(y_true) == cancer_type
            
            if not np.any(type_mask):
                continue
                
            # Get predictions for these samples
            preds = y_pred_proba[type_mask]
            
            # Get top and second predictions
            top_indices = np.argmax(preds, axis=1)
            
            # Create a mask to ignore the top prediction for finding the second best
            mask = np.zeros_like(preds)
            mask[np.arange(preds.shape[0]), top_indices] = 1
            masked_preds = np.ma.array(preds, mask=mask)
            second_indices = np.argmax(masked_preds, axis=1)
            
            # Check if predictions are correct
            top_correct = (top_indices == np.where(cancer_types == cancer_type)[0][0]).sum()
            top2_correct = top_correct + ((second_indices == np.where(cancer_types == cancer_type)[0][0]).sum())
            
            # Calculate accuracies
            top_acc = top_correct / type_mask.sum()
            top2_acc = top2_correct / type_mask.sum()
            
            results[cancer_type] = {
                'top': top_acc,
                'top2': top2_acc,
                'count': type_mask.sum()
            }
    
    # Convert results to DataFrame for plotting
    df = pd.DataFrame([{
        'Cancer_Type': ct,
        'Top_Accuracy': res['top'],
        'Top2_Accuracy': res['top2'],
        'Count': res['count']
    } for ct, res in results.items()])
    
    # Sort by top accuracy (descending)
    df = df.sort_values('Top_Accuracy', ascending=False)
    
    # Plot stacked bar chart
    bar_width = 0.8
    bottom_bars = ax.bar(df['Cancer_Type'], df['Top_Accuracy'], width=bar_width, 
                         color='lightblue', label='Top Prediction')
    
    # Calculate the additional height for top 2 accuracy
    top2_additional = df['Top2_Accuracy'] - df['Top_Accuracy']
    top_bars = ax.bar(df['Cancer_Type'], top2_additional, width=bar_width, 
                      bottom=df['Top_Accuracy'], color='steelblue', label='Top 2 Predictions')
    
    # Calculate confidence intervals using Clopper-Pearson
    ci_low_top = []
    ci_high_top = []
    ci_low_top2 = []
    ci_high_top2 = []
    
    for _, row in df.iterrows():
        top_correct = int(row['Top_Accuracy'] * row['Count'])
        top2_correct = int(row['Top2_Accuracy'] * row['Count'])
        count = int(row['Count'])
        
        # Calculate CIs for top prediction
        ci_low, ci_high = smp.proportion_confint(top_correct, count, alpha=0.05, method='beta')
        ci_low_top.append(ci_low)
        ci_high_top.append(ci_high)
        
        # Calculate CIs for top 2 predictions
        ci_low, ci_high = smp.proportion_confint(top2_correct, count, alpha=0.05, method='beta')
        ci_low_top2.append(ci_low)
        ci_high_top2.append(ci_high)
    
    # Add error bars
    ax.errorbar(
        df['Cancer_Type'], 
        df['Top_Accuracy'],
        yerr=[df['Top_Accuracy'] - ci_low_top, ci_high_top - df['Top_Accuracy']],
        fmt='none', 
        color='black', 
        capsize=5
    )
    
    ax.errorbar(
        df['Cancer_Type'], 
        df['Top2_Accuracy'],
        yerr=[df['Top2_Accuracy'] - ci_low_top2, ci_high_top2 - df['Top2_Accuracy']],
        fmt='none', 
        color='black', 
        capsize=5
    )
    
    # Annotate bars with percentage
    for i, (index, row) in enumerate(df.iterrows()):
        # Annotate top prediction accuracy
        ax.text(i, row['Top_Accuracy'] / 2, f"{row['Top_Accuracy']:.0%}", 
                ha='center', va='center', color='black', fontweight='bold')
        
        # Annotate top 2 prediction accuracy
        ax.text(i, row['Top_Accuracy'] + top2_additional[i] / 2, f"{top2_additional[i]:.0%}", 
                ha='center', va='center', color='white', fontweight='bold')
        
        # Annotate total accuracy on top
        ax.text(i, row['Top2_Accuracy'] + 0.02, f"{row['Top2_Accuracy']:.0%}", 
                ha='center', va='bottom', color='black')
    
    # Add labels and title
    ax.set_xlabel('Cancer Type')
    ax.set_ylabel('Accuracy of Prediction (%)')
    
    # Update title with model types and standardization status if provided
    final_title = "Identification of cancer type by supervised machine learning for patients classified as positive by CancerSEEK"
    if detection_model and localization_model:
        final_title = f"Identification of cancer type using {localization_model} for patients classified as positive by CancerSEEK ({detection_model})"
    if standardized is not None:
        std_text = " (Standardized)" if standardized else " (Not Standardized)"
        final_title += std_text
    ax.set_title(final_title)
    
    # Create custom legend
    legend_elements = [
        Patch(facecolor='lightblue', label='Top Prediction'),
        Patch(facecolor='steelblue', label='Additional from Top 2 Predictions')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set y-axis limit
    ax.set_ylim([0, 1.05])
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add percentage labels to y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

def plot_multiple_roc_curves(data_files, labels=None, colors=None, figsize=(12, 10), title="Comparison of ROC Curves", save_path=None):
    """
    Plot multiple ROC curves for comparison.
    
    Parameters:
    -----------
    data_files : list
        List of file paths to pickled ROC data files generated by plot_roc_curve
    labels : list, optional
        List of labels for each ROC curve (for legend)
    colors : list, optional
        List of colors for each ROC curve
    figsize : tuple, optional
        Figure size
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the comparison plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Default colors if not provided
    if colors is None:
        colors = plt.cm.tab10.colors
    
    # Load and plot each ROC curve
    for i, data_file in enumerate(data_files):
        # Load data
        with open(data_file, 'rb') as f:
            plot_data = pickle.load(f)
        
        # Extract data
        fpr = plot_data['fpr']
        tpr = plot_data['tpr']
        auc_value = plot_data['auc']
        
        # Get sensitivity at target specificity
        sens_results = plot_data['sensitivity_at_target_spec']
        
        # Find the point on the curve closest to the target specificity
        target_specificity = sens_results['specificity']
        target_fpr = 1 - target_specificity
        idx = np.argmin(np.abs(fpr - target_fpr))
        spec_point_x = fpr[idx]
        spec_point_y = tpr[idx]
        
        # Determine label
        curve_label = labels[i] if labels is not None and i < len(labels) else f"Model {i+1}"
        label = f"{curve_label} (AUC = {auc_value:.3f})"
        
        # Plot ROC curve
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, lw=2, color=color, label=label)
        
        # Add dot at specificity point
        ax.plot(spec_point_x, spec_point_y, 'o', color=color, markersize=8)
        
        # Add vertical error bars for confidence intervals
        ax.errorbar(spec_point_x, spec_point_y, 
                    yerr=[[spec_point_y - sens_results['ci_low']], 
                          [sens_results['ci_high'] - spec_point_y]],
                    fmt='none', color=color, capsize=5, alpha=0.7)
        
        # Add annotation for sensitivity at target specificity
        ax.annotate(f"{sens_results['sensitivity']:.3f} ({sens_results['ci_low']:.3f}-{sens_results['ci_high']:.3f})",
                    xy=(spec_point_x, spec_point_y),
                    xytext=(spec_point_x + 0.05, spec_point_y - 0.05),
                    arrowprops=dict(arrowstyle='->', color=color),
                    color=color,
                    fontsize=9)
    
    # Add random chance line
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Add labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc="lower right")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig, ax

def plot_tissue_localization_confusion_matrix(y_true=None, y_pred=None, cm=None, cancer_types=None, figsize=(12, 10),
                                             detection_model=None, localization_model=None, 
                                             normalize=None, save_path=None, standardized=None):
    """
    Plot confusion matrix for tissue localization. 
    Can accept either true/predicted labels (y_true, y_pred) or a pre-computed confusion matrix (cm).
    
    Parameters:
    -----------
    y_true : array-like, optional
        True cancer types. Required if 'cm' is not provided.
    y_pred : array-like, optional
        Predicted cancer types. Required if 'cm' is not provided.
    cm : pd.DataFrame, optional
        Pre-computed confusion matrix (rows=true labels, columns=predicted labels). 
        If provided, y_true and y_pred are ignored.
    cancer_types : list, optional
        List of cancer type names. If None, inferred from labels or 'cm'.
    figsize : tuple, optional
        Figure size
    detection_model : str, optional
        Type of model used for detection (e.g., 'LR', 'XGB')
    localization_model : str, optional
        Type of model used for tissue localization (e.g., 'RF', 'XGB')
    normalize : str, optional
        Normalization method ('true', 'pred', 'all'), or None for raw counts/pre-normalized input.
        This affects display format and title suffix.
    save_path : str, optional
        Path to save the confusion matrix plot
    standardized : bool, optional
        Indicates if features were standardized (default: None)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    metrics : dict or None
        Dictionary containing accuracy and F1 scores (only if calculated from y_true/y_pred).
    """
    metrics = None  # Initialize metrics as None

    if cm is None:
        # Calculate confusion matrix from y_true and y_pred
        if y_true is None or y_pred is None:
            raise ValueError("Either 'cm' or both 'y_true' and 'y_pred' must be provided.")
            
        # Check that the inputs have the same length
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements but y_pred has {len(y_pred)} elements")
            
        # If cancer_types not provided, infer from y_true and y_pred
        if cancer_types is None:
            cancer_types = sorted(np.union1d(np.unique(y_true), np.unique(y_pred)))
        
        # Print debug info
        print(f"Number of samples: {len(y_true)}")
        print(f"Cancer types for matrix: {cancer_types}")
        
        # Create numeric mappings for cancer types
        type_to_idx = {cancer_type: i for i, cancer_type in enumerate(cancer_types)}
        
        # Convert cancer types to class indices for metrics calculation
        y_true_numeric = np.array([type_to_idx.get(t, -1) for t in y_true])
        y_pred_numeric = np.array([type_to_idx.get(p, -1) for p in y_pred])
        
        # Filter out any samples with unknown types (-1 index)
        valid_samples = (y_true_numeric >= 0) & (y_pred_numeric >= 0)
        
        # Check if we have valid samples
        if not np.any(valid_samples):
            raise ValueError("No valid samples found after filtering. Cannot create confusion matrix.")
        
        # Apply filtering
        y_true_filtered = y_true_numeric[valid_samples]
        y_pred_filtered = y_pred_numeric[valid_samples]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
        macro_f1 = f1_score(y_true_filtered, y_pred_filtered, average='macro')
        micro_f1 = f1_score(y_true_filtered, y_pred_filtered, average='micro')
        
        # Compute confusion matrix
        cm_calculated = confusion_matrix(y_true_filtered, y_pred_filtered, labels=range(len(cancer_types)))
        
        # Convert to DataFrame for consistency
        cm = pd.DataFrame(cm_calculated, index=cancer_types, columns=cancer_types)
        
        # Store metrics
        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1
        }

        # Normalize if requested (calculation)
        if normalize == 'true':
            cm_normalized = cm.astype('float').values / cm.sum(axis=1).values[:, np.newaxis]
            cm_display = pd.DataFrame(cm_normalized, index=cm.index, columns=cm.columns)
            fmt = '.2f'
            vmin, vmax = 0, 1
            cmap = 'Blues'
            title_suffix = " (Normalized by True Label)"
        elif normalize == 'pred':
            cm_normalized = cm.astype('float').values / cm.sum(axis=0).values[np.newaxis, :]
            cm_display = pd.DataFrame(cm_normalized, index=cm.index, columns=cm.columns)
            fmt = '.2f'
            vmin, vmax = 0, 1
            cmap = 'Blues'
            title_suffix = " (Normalized by Prediction)"
        elif normalize == 'all':
            cm_normalized = cm.astype('float').values / cm.sum().sum()
            cm_display = pd.DataFrame(cm_normalized, index=cm.index, columns=cm.columns)
            fmt = '.3f'
            vmin, vmax = 0, 1
            cmap = 'Blues'
            title_suffix = " (Normalized)"
        else:
            cm_display = cm
            fmt = 'd'
            vmin, vmax = None, None
            cmap = 'Blues'
            title_suffix = ""

    else:
        # Use the provided confusion matrix DataFrame
        if not isinstance(cm, pd.DataFrame):
            raise TypeError("'cm' must be a pandas DataFrame.")
        
        # Infer cancer types from DataFrame index/columns if not provided
        if cancer_types is None:
            cancer_types = list(cm.index)
            # Basic check that index and columns match
            if list(cm.index) != list(cm.columns):
                 print("Warning: Confusion matrix index and columns do not match. Using index for labels.")
        
        print(f"Using pre-computed confusion matrix with types: {cancer_types}")
        
        # Determine display format based on 'normalize' argument and data type
        # Assume the input 'cm' might already be normalized or raw counts
        cm_display = cm 
        
        if normalize == 'true':
            # Assume cm is already normalized by row (true) or represents fractions
            fmt = '.2f' 
            vmin, vmax = 0, 1
            cmap = 'Blues'
            title_suffix = " (Normalized by True Label)"
        elif normalize == 'pred':
            # Assume cm is already normalized by col (pred) or represents fractions
            fmt = '.2f' 
            vmin, vmax = 0, 1
            cmap = 'Blues'
            title_suffix = " (Normalized by Prediction)"
        elif normalize == 'all':
            # Assume cm is already normalized by all or represents fractions
            fmt = '.3f'
            vmin, vmax = 0, 1
            cmap = 'Blues'
            title_suffix = " (Normalized)"
        else:
             # Check if data looks like counts (integers) or fractions/percentages
            if cm_display.values.dtype == 'int' or (cm_display.values.max() > 1.1): # Heuristic for counts
                 fmt = 'd'
                 vmin, vmax = None, None
                 cmap = 'Blues'
                 title_suffix = " (Counts)"
            else: # Assume fractions/percentages
                 fmt = '.2f' 
                 vmin, vmax = 0, 1
                 cmap = 'Blues'
                 title_suffix = " (Pre-computed Values)"
            
            normalize = None # Ensure normalize flag matches data type assumption

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=cancer_types, yticklabels=cancer_types,
                vmin=vmin, vmax=vmax, ax=ax, cbar=normalize is not None) # Only show cbar if normalized
    
    # Customize axes
    ax.set_xlabel('Predicted Cancer Type')
    ax.set_ylabel('True Cancer Type')
    
    # Update title with model types and standardization status if provided
    final_title = "Tissue Localization Confusion Matrix"
    if localization_model: # Removed detection model as it's less relevant here
        final_title = f"Tissue Localization Confusion Matrix ({localization_model} model)"
    
    final_title += title_suffix
    
    if standardized is not None:
        std_text = " (Standardized)" if standardized else " (Not Standardized)"
        final_title += std_text

    # Create a subtitle with metrics if they were calculated
    if metrics:
        metrics_text = f"Accuracy: {metrics['accuracy']:.3f}, Macro F1: {metrics['macro_f1']:.3f}, Micro F1: {metrics['micro_f1']:.3f}"
        ax.set_title(f"{final_title}\\n{metrics_text}", pad=15)
    else:
        ax.set_title(final_title, pad=15)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        print(f"Saving confusion matrix plot to: {save_path}")
        plt.savefig(save_path, bbox_inches='tight')
        print("Plot saved.")
    
    return fig, ax, metrics

def calculate_sensitivity_by_subtype_and_stage(y_true, y_score, subtypes, stages, target_specificity=0.99):
    """
    Calculate sensitivity at a target specificity for each cancer subtype and stage combination.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (1 for cancer, 0 for normal)
    y_score : array-like
        Target scores (probability estimates of the positive class)
    subtypes : array-like
        Cancer subtypes for each sample (empty string or NaN for normal samples)
    stages : array-like
        Cancer stages for each sample (empty string or NaN for normal samples)
    target_specificity : float, optional
        Target specificity value (default: 0.99)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with sensitivity and confidence intervals for each subtype-stage combination
    """
    # Find threshold for target specificity on the entire dataset
    result_all = calculate_sensitivity_at_specificity(y_true, y_score, target_specificity)
    threshold = result_all['threshold']
    
    # Initialize results list
    results = []
    
    # Get unique cancer types (excluding Normal)
    unique_subtypes = [s for s in np.unique(subtypes) if s and not pd.isna(s) and s != 'Normal']
    
    # Get unique cancer stages
    unique_stages = []
    for stage in ['I', 'II', 'III', 'IV']:
        if stage in stages.unique():
            unique_stages.append(stage)
    
    # Iterate through each subtype
    for subtype in unique_subtypes:
        # Calculate overall sensitivity for this subtype
        subtype_mask = (subtypes == subtype)
        
        if np.sum(subtype_mask) == 0:
            continue
        
        # Calculate TP and P for this subtype
        tp_subtype = np.sum((subtype_mask) & (y_score >= threshold))
        p_subtype = np.sum(subtype_mask)
        
        # Calculate sensitivity for this subtype
        if p_subtype > 0:
            sensitivity_subtype = tp_subtype / p_subtype
            # Calculate Clopper-Pearson confidence interval
            ci_low_subtype, ci_high_subtype = smp.proportion_confint(tp_subtype, p_subtype, alpha=0.05, method='beta')
        else:
            sensitivity_subtype = np.nan
            ci_low_subtype, ci_high_subtype = np.nan, np.nan
        
        # Add overall subtype result
        results.append({
            'Cancer_Type': subtype,
            'Stage': 'All',
            'Sensitivity': sensitivity_subtype,
            'CI_Low': ci_low_subtype,
            'CI_High': ci_high_subtype,
            'True_Positives': tp_subtype,
            'Total_Positives': p_subtype,
            'Specificity': result_all['specificity'],
            'Specificity_Rounded': result_all['specificity_rounded']
        })
        
        # Calculate sensitivity for each stage within this subtype
        for stage in unique_stages:
            # Create mask for this subtype and stage
            stage_mask = (subtypes == subtype) & (stages == stage)
            
            if np.sum(stage_mask) == 0:
                continue
            
            # Calculate TP and P for this stage
            tp_stage = np.sum((stage_mask) & (y_score >= threshold))
            p_stage = np.sum(stage_mask)
            
            # Calculate sensitivity for this stage
            if p_stage > 0:
                sensitivity_stage = tp_stage / p_stage
                # Calculate Clopper-Pearson confidence interval
                ci_low_stage, ci_high_stage = smp.proportion_confint(tp_stage, p_stage, alpha=0.05, method='beta')
            else:
                sensitivity_stage = np.nan
                ci_low_stage, ci_high_stage = np.nan, np.nan
            
            # Add stage result
            results.append({
                'Cancer_Type': subtype,
                'Stage': stage,
                'Sensitivity': sensitivity_stage,
                'CI_Low': ci_low_stage,
                'CI_High': ci_high_stage,
                'True_Positives': tp_stage,
                'Total_Positives': p_stage,
                'Specificity': result_all['specificity'],
                'Specificity_Rounded': result_all['specificity_rounded']
            })
    
    # Convert to DataFrame
    return pd.DataFrame(results)

def plot_sensitivity_by_subtype_and_stage(sensitivity_df, figsize=(15, 12), model_type=None, standardized=None, ncols=3):
    """
    Plot faceted sensitivity charts showing sensitivity by stage for each cancer subtype.
    
    Parameters:
    -----------
    sensitivity_df : pd.DataFrame
        DataFrame with sensitivity data from calculate_sensitivity_by_subtype_and_stage
    figsize : tuple, optional
        Figure size
    model_type : str, optional
        Type of model used for detection (e.g., 'LR', 'XGB')
    standardized : bool, optional
        Indicates if features were standardized (default: None)
    ncols : int, optional
        Number of columns in the facet grid (default: 3)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    axes : numpy.ndarray
        Array of axes objects
    spec_percent : int
        Specificity percentage (for filename)
    """
    # Get unique cancer types
    cancer_types = sensitivity_df['Cancer_Type'].unique()
    cancer_types = [ct for ct in cancer_types if ct != 'All Cancers']
    
    # Get the specificity value for display (rounded to 3 sig figs)
    spec_rounded = sensitivity_df['Specificity_Rounded'].iloc[0]
    spec_percent = int(spec_rounded * 1000)
    
    # Calculate number of rows needed
    nrows = (len(cancer_types) + ncols - 1) // ncols
    
    # Create figure and axes grid
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    
    # Define consistent colors for stages
    stage_colors = {
        'I': 'lightgreen',
        'II': 'yellowgreen',
        'III': 'orange',
        'IV': 'tomato'
    }
    
    # Define stage order
    stage_order = ['I', 'II', 'III', 'IV']
    
    # Plot each cancer type in its own subplot
    for i, cancer_type in enumerate(cancer_types):
        # Calculate subplot position
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        # Filter data for this cancer type
        df_subtype = sensitivity_df[sensitivity_df['Cancer_Type'] == cancer_type]
        df_stages = df_subtype[df_subtype['Stage'] != 'All']
        
        # Create a list of stages present in this cancer type
        available_stages = []
        for stage in stage_order:
            if stage in df_stages['Stage'].values:
                available_stages.append(stage)
        
        # Skip if no stages available
        if len(available_stages) == 0:
            ax.text(0.5, 0.5, f"No stage data for\n{cancer_type}", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Filter and order by available stages
        df_plot = df_stages[df_stages['Stage'].isin(available_stages)]
        df_plot = df_plot.set_index('Stage').loc[available_stages].reset_index()
        
        # Get colors for the bars
        bar_colors = [stage_colors[stage] for stage in df_plot['Stage']]
        
        # Plot bar chart
        bars = ax.bar(df_plot['Stage'], df_plot['Sensitivity'], color=bar_colors)
        
        # Add error bars
        ax.errorbar(
            df_plot['Stage'], 
            df_plot['Sensitivity'],
            yerr=[df_plot['Sensitivity'] - df_plot['CI_Low'], 
                  df_plot['CI_High'] - df_plot['Sensitivity']],
            fmt='none', 
            color='black', 
            capsize=5
        )
        
        # Add sensitivity values as text on each bar
        for j, (_, row) in enumerate(df_plot.iterrows()):
            # Show sensitivity as percentage
            ax.text(j, row['Sensitivity'] / 2, f"{row['Sensitivity']*100:.0f}%", 
                    ha='center', va='center', color='black', fontweight='bold')
            
            # Also add the total count as smaller text just below the sensitivity value
            ax.text(j, row['Sensitivity'] / 2 - 0.05, f"({row['True_Positives']}/{row['Total_Positives']})", 
                    ha='center', va='center', color='black', fontsize=8)
        
        # Set subplot title to cancer type
        ax.set_title(cancer_type)
        
        # Set y-axis limit
        ax.set_ylim([0, 1.05])
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        # Only show y-axis labels for leftmost subplots
        if col == 0:
            ax.set_ylabel('Sensitivity')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Hide any unused subplots
    for i in range(len(cancer_types), nrows * ncols):
        row = i // ncols
        col = i % ncols
        fig.delaxes(axes[row, col])
    
    # Add a common title for the entire figure
    fig_title = f"Sensitivity by Cancer Type and Stage at {spec_percent/10:.1f}% Specificity"
    if model_type:
        fig_title = f"Sensitivity by Cancer Type and Stage at {spec_percent/10:.1f}% Specificity ({model_type})"
    if standardized is not None:
        std_text = " (Standardized)" if standardized else " (Not Standardized)"
        fig_title += std_text
    
    fig.suptitle(fig_title, fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    
    return fig, axes, spec_percent
