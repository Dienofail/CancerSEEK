import pandas as pd
import numpy as np

# Base path for all data files
BASE_PATH = "/Users/alvinshi/Library/CloudStorage/Dropbox/Interview_prep/Exact_Sciences/CancerSEEK/data"

def clean_column_names(df):
    """
    Clean column names by replacing spaces with underscores, removing '#' symbols,
    and removing measurement units.
    """
    # Create a mapping for column renaming
    rename_map = {}
    for col in df.columns:
        # Replace spaces with underscores and remove '#' symbols
        new_col = col.replace(' ', '_').replace('#', '').replace('*', '')
        
        # Remove measurement units if they exist
        for unit in ['(U/ml)', '(pg/ml)', '(ng/mL)', '(nM)', '(%)', '_(ng/ml)']:
            new_col = new_col.replace(unit, '')
        
        # Ensure specific column standardization for joins
        if 'Patient ID' in col:
            new_col = 'Patient_ID'
        elif 'Sample ID' in col:
            new_col = 'Sample_ID'
        elif 'Plasma sample ID' in col or 'Plasma_sample_ID' in col:
            new_col = 'Sample_ID'
        
        # Replace mean_allele_frequency with MAF
        new_col = new_col.replace('mean_allele_frequency', 'MAF')
        # Replace allele_frequency with AF
        new_col = new_col.replace('allele_frequency', 'AF')
        
        # Remove trailing underscores
        new_col = new_col.rstrip('_')
        
        rename_map[col] = new_col
    
    # Apply the renaming
    return df.rename(columns=rename_map)


def convert_percentages_to_fractions(df):
    """
    Convert percentage values to fractions (0-1 range) in specified columns.
    """
    for col in df.columns:
        # Handle string percentage values
        if df[col].dtype == 'object':
            # Check if column contains percentage values
            if df[col].astype(str).str.contains('%').any():
                df[col] = df[col].astype(str).str.rstrip('%').astype('float') / 100
        
        # Also look for columns with "%" in the name which might indicate percentages
        if '%' in col or 'percent' in col.lower():
            try:
                # If column is already numeric
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check if values are likely percentages (>1)
                    if df[col].median() > 1:
                        df[col] = df[col] / 100
                else:
                    # Convert string to numeric first
                    df[col] = df[col].astype(str).str.rstrip('%').astype('float') / 100
            except:
                pass  # Not a numeric column, skip conversion
    
    return df


def load_s1_primers():
    """
    Load and process the primers data from CancerSeek_s1_primers.csv
    """
    file_path = f"{BASE_PATH}/CancerSeek_s1_primers.csv"
    df = pd.read_csv(file_path)
    
    # Clean column names
    df = clean_column_names(df)
    
    return df


def load_s2_mutations():
    """
    Load and process the mutations data from CancerSEEK_s2_mutations.csv
    """
    file_path = f"{BASE_PATH}/CancerSEEK_s2_mutations.csv"
    df = pd.read_csv(file_path)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Convert percentages to fractions
    if 'Mutant_MAF' in df.columns:
        # Check if the column is already numeric
        if df['Mutant_MAF'].dtype == 'object':
            # If strings, process them
            df['Mutant_MAF'] = df['Mutant_MAF'].astype(str).str.rstrip('%').astype(float) / 100
        else:
            # If already numeric, just divide by 100 if values are likely percentages (>1)
            if df['Mutant_MAF'].median() > 1:  # Simple heuristic to check if values are percentages
                df['Mutant_MAF'] = df['Mutant_MAF'] / 100
    
    # Replace spaces in IDs with underscores
    if 'Patient_ID' in df.columns:
        df['Patient_ID'] = df['Patient_ID'].str.replace(' ', '_')
    if 'Sample_ID' in df.columns:
        df['Sample_ID'] = df['Sample_ID'].str.replace(' ', '_')
    
    return df


def load_s3_protein_biomarkers():
    """
    Load and process the protein biomarkers data from CancerSEEK_s3_protein_biomarkers_included.csv
    """
    file_path = f"{BASE_PATH}/CancerSEEK_s3_protein_biomarkers_included.csv"
    df = pd.read_csv(file_path)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Convert Yes/No to boolean for clarity
    for col in df.columns:
        if df[col].dtype == 'object':
            if set(df[col].unique()).issubset({'Yes', 'No'}):
                df[col] = df[col].map({'Yes': True, 'No': False})
    
    return df


def load_s4_clinical_characteristics():
    """
    Load and process the clinical characteristics data from CancerSEEK_s4_clinical_characteristics.csv
    """
    file_path = f"{BASE_PATH}/CancerSEEK_s4_clinical_characteristics.csv"
    df = pd.read_csv(file_path)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Replace spaces in IDs with underscores
    if 'Patient_ID' in df.columns:
        df['Patient_ID'] = df['Patient_ID'].str.replace(' ', '_')
    if 'Sample_ID' in df.columns:
        df['Sample_ID'] = df['Sample_ID'].str.replace(' ', '_')
    if 'Plasma_sample_ID' in df.columns:
        df['Sample_ID'] = df['Plasma_sample_ID'].str.replace(' ', '_')
        df = df.drop(columns=['Plasma_sample_ID'])
    if 'Primary_tumor_sample_ID' in df.columns:
        df['Primary_tumor_sample_ID'] = df['Primary_tumor_sample_ID'].str.replace(' ', '_')
    
    # Convert test result to boolean
    if 'CancerSEEK_Test_Result' in df.columns:
        df['CancerSEEK_Test_Result'] = df['CancerSEEK_Test_Result'].map({'Positive': True, 'Negative': False})
    
    return df


def load_s5_mutations():
    """
    Load and process the plasma mutations data from CancerSEEK_s5_mutations.csv
    """
    file_path = f"{BASE_PATH}/CancerSEEK_s5_mutations.csv"
    df = pd.read_csv(file_path)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Replace spaces in IDs with underscores
    if 'Patient_ID' in df.columns:
        df['Patient_ID'] = df['Patient_ID'].str.replace(' ', '_')
    if 'Sample_ID' in df.columns:
        df['Sample_ID'] = df['Sample_ID'].str.replace(' ', '_')
    
    # Convert percentage columns to fractions
    percentage_cols = ['Mutant_AF']
    for col in percentage_cols:
        if col in df.columns:
            # Check if the column is already numeric
            if df[col].dtype == 'object':
                # If strings, process them
                mask = df[col].notna()
                df.loc[mask, col] = df.loc[mask, col].astype(str).str.rstrip('%').astype(float) / 100
            else:
                # If already numeric, just divide by 100 if values are likely percentages (>1)
                if df[col].median() > 1:  # Simple heuristic to check if values are percentages
                    df[col] = df[col] / 100
    
    # Rename specific columns for easier reference
    if 'Ω_score' in df.columns:
        df = df.rename(columns={'Ω_score': 'omega_score'})
    
    # Fix the fragments per mL plasma column name
    for col in df.columns:
        if 'Mutant_fragments/mL_plasma' in col:
            df = df.rename(columns={col: 'per_mL_plasma'})
    
    # Impute NA values in specified numerical columns
    columns_to_impute = ['omega_score', 'Mutant_AF', 'per_mL_plasma']
    
    # Identify non-cancer samples
    if 'Type' in df.columns:
        normal_type = 'Normal'
        non_cancer_samples = df[df['Type'] == normal_type]
    elif 'Tumor_type' in df.columns:
        normal_type = 'Normal'
        non_cancer_samples = df[df['Tumor_type'] == normal_type]
    else:
        # No column to identify normal samples
        non_cancer_samples = pd.DataFrame()
    
    if not non_cancer_samples.empty:
        # Impute each column
        for col in columns_to_impute:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Get valid non-cancer values (explicitly exclude NaN values)
                valid_values = non_cancer_samples[col][~pd.isna(non_cancer_samples[col])]
                
                # Only calculate mean if we have valid values
                if len(valid_values) > 0:
                    mean_value = valid_values.mean()
                    # Fill NA values with calculated mean
                    df[col] = df[col].fillna(mean_value)
    
    return df


def load_s6_protein_conc():
    """
    Load and process the protein concentration data from CancerSEEK_s6_protein_conc.csv
    """
    file_path = f"{BASE_PATH}/CancerSEEK_s6_protein_conc.csv"
    df = pd.read_csv(file_path)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Replace spaces in IDs with underscores
    if 'Patient_ID' in df.columns:
        df['Patient_ID'] = df['Patient_ID'].str.replace(' ', '_')
    if 'Sample_ID' in df.columns:
        df['Sample_ID'] = df['Sample_ID'].str.replace(' ', '_')
    
    # Handle values with asterisks and commas by removing them and converting to numeric
    non_protein_cols = ['Patient_ID', 'Sample_ID', 'CancerSEEK_Test_Result', 'Cancer_Type', 'Type']
    for col in df.columns:
        # Skip ID columns and non-numeric columns
        if col in non_protein_cols:
            continue
            
        if df[col].dtype == 'object':
            # First remove commas from all string values
            df[col] = df[col].astype(str).str.replace(',', '')
            
            # Then remove asterisks if present
            if df[col].astype(str).str.contains(r'\*').any():
                df[col] = df[col].astype(str).str.replace(r'\*', '', regex=True)
            
            # Try to convert to float after cleaning
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                # If conversion fails, keep the column as is
                pass
    
    # Convert test result to boolean
    if 'CancerSEEK_Test_Result' in df.columns:
        df['CancerSEEK_Test_Result'] = df['CancerSEEK_Test_Result'].map({'Positive': True, 'Negative': False})
    
    # Impute NA values in numeric columns with mean of non-cancer samples
    protein_cols = [col for col in df.columns if col not in non_protein_cols]
    
    # Only proceed with imputation if Type column exists
    if 'Type' in df.columns:
        normal_type = 'Normal'
        non_cancer_samples = df[df['Type'] == normal_type]
    elif 'Tumor_type' in df.columns:
        normal_type = 'Normal'
        non_cancer_samples = df[df['Tumor_type'] == normal_type]
    else:
        # No column to identify normal samples
        non_cancer_samples = pd.DataFrame()
    
    if not non_cancer_samples.empty:
        # Calculate means for non-cancer samples for each protein column
        for col in protein_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Get valid non-cancer values (explicitly exclude NaN values)
                valid_values = non_cancer_samples[col][~pd.isna(non_cancer_samples[col])]
                
                # Only calculate mean if we have valid values
                if len(valid_values) > 0:
                    mean_value = valid_values.mean()
                    # Fill NA values with calculated mean
                    df[col] = df[col].fillna(mean_value)
    
    return df


def load_s7_concordance_mutations():
    """
    Load and process the concordance mutations data from CancerSEEK_s7_concordance_mutations.csv
    """
    file_path = f"{BASE_PATH}/CancerSEEK_s7_concordance_mutations.csv"
    df = pd.read_csv(file_path)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Process specific columns
    if 'Mutant_AF_(%)' in df.columns:
        # Check if the column is already numeric
        if df['Mutant_AF_(%)'].dtype == 'object':
            df['Mutant_AF'] = df['Mutant_AF_(%)'].astype(str).str.replace('%', '').astype(float) / 100
        else:
            df['Mutant_AF'] = df['Mutant_AF_(%)'] / 100
        df = df.drop(columns=['Mutant_AF_(%)'])
    
    # Replace spaces in Patient_ID and Sample_ID values with underscores
    if 'Patient_ID' in df.columns:
        df['Patient_ID'] = df['Patient_ID'].str.replace(' ', '_')
    if 'Sample_ID' in df.columns:
        df['Sample_ID'] = df['Sample_ID'].str.replace(' ', '_')
    
    return df


def load_s8_cancertype_localization():
    """
    Load and process the cancer type localization data from CancerSEEK_s8_cancertype_localization.csv
    """
    file_path = f"{BASE_PATH}/CancerSEEK_s8_cancertype_localization.csv"
    df = pd.read_csv(file_path)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Replace spaces in IDs with underscores
    if 'Patient_ID' in df.columns:
        df['Patient_ID'] = df['Patient_ID'].str.replace(' ', '_')
    if 'Sample_ID' in df.columns:
        df['Sample_ID'] = df['Sample_ID'].str.replace(' ', '_')
    
    # Convert probability columns to float
    probability_cols = [col for col in df.columns if 'Probability' in col]
    for col in probability_cols:
        df[col] = df[col].astype(float)
    
    return df


def load_s9_logistic_regression():
    """
    Load and process the logistic regression model coefficients from CancerSEEK_s9_logistic_regression_model_coefficients.csv
    """
    file_path = f"{BASE_PATH}/CancerSEEK_s9_logistic_regression_model_coefficients.csv"
    df = pd.read_csv(file_path)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Process scientific notation - no need for explicit handling as pandas will load them correctly
    # Just ensure they're numeric
    numeric_cols = ['Logistic_Regression_Coefficient', 'Importance_Score']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    return df


def load_s10_confusion_matrix():
    """
    Load and process the confusion matrix from CancerSEEK_s10_confusion_matrix.csv
    """
    file_path = f"{BASE_PATH}/CancerSEEK_s10_confusion_matrix.csv"
    df = pd.read_csv(file_path, index_col=0)
    
    # Convert percentages to fractions
    for col in df.columns:
        if df[col].dtype == 'object':
            # If strings, convert percentage strings to float
            df[col] = df[col].astype(str).str.rstrip('%').astype('float') / 100
        else:
            # If already numeric, check if values are likely percentages
            if df[col].median() > 1:
                df[col] = df[col] / 100
    
    # Rename columns to indicate actual cancer types
    df.columns = ['actual_' + col for col in df.columns]
    
    # Rename index to indicate predicted cancer types
    df.index = ['predicted_' + idx for idx in df.index]
    
    return df


def load_all_supplemental_files():
    """
    Load all supplemental files and return them in a dictionary.
    Keys are the file names without the 'CancerSEEK_' prefix and '.csv' ending.
    """
    supplemental_data = {
        's1_primers': load_s1_primers(),
        's2_mutations': load_s2_mutations(),
        's3_protein_biomarkers': load_s3_protein_biomarkers(),
        's4_clinical_characteristics': load_s4_clinical_characteristics(),
        's5_mutations': load_s5_mutations(),
        's6_protein_conc': load_s6_protein_conc(),
        's7_concordance_mutations': load_s7_concordance_mutations(),
        's8_cancertype_localization': load_s8_cancertype_localization(),
        's9_logistic_regression': load_s9_logistic_regression(),
        's10_confusion_matrix': load_s10_confusion_matrix()
    }
    
    return supplemental_data


if __name__ == "__main__":
    # Example usage
    all_data = load_all_supplemental_files()
    print("Loaded data keys:", all_data.keys())
    
    # Print sample from each dataset
    for key, df in all_data.items():
        print(f"\n{key} sample:")
        print(df.head(2))
