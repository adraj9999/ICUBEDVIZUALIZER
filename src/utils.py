import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features from the raw dataset. This function is designed to work
    even after some columns have been removed.
    """
    features = df.copy()

    # --- Feature Creation ---
    # To avoid division by zero, replace 0s in divisors with NaN for safe calculation
    if 'Adult Population' in features.columns:
        features['Adult Population'] = features['Adult Population'].replace(0, np.nan)

    # Calculate ratios and utilization metrics if base columns exist
    if 'Population 65+' in features.columns and 'Adult Population' in features.columns:
        features['senior_ratio'] = features['Population 65+'] / features['Adult Population']
    
    # The other leaky/removed features are not created here as their base columns are gone.
    
    # --- Imputation ---
    # Fill any NaNs created during calculations with the column's median
    numeric_cols = features.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if features[col].isnull().any():
            median_val = features[col].median()
            features[col] = features[col].fillna(median_val)
            
    return features


def prepare_data_for_model(df: pd.DataFrame, model_feature_names: list[str]) -> pd.DataFrame:
    """
    Prepares a feature-engineered dataframe for prediction by a specific model.
    1. Cleans column names to be alphanumeric with underscores.
    2. Ensures the dataframe has the exact features the model was trained on.
    """
    data_to_prepare = df.copy()
    
    # Clean column names to match the format used during training
    data_to_prepare.columns = ["".join(c if c.isalnum() else "_" for c in str(col)) for col in data_to_prepare.columns]
    
    # Create a new dataframe containing only the features the model expects
    model_input_df = pd.DataFrame(index=data_to_prepare.index)
    
    for feature in model_feature_names:
        if feature in data_to_prepare.columns:
            model_input_df[feature] = data_to_prepare[feature]
        else:
            # If a required feature is missing, add it and fill with 0 as a safe default
            model_input_df[feature] = 0
            
    return model_input_df
