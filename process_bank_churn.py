import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple, Dict, Any


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not useful for modeling.
    
    Args:
        df: Raw input DataFrame.
    
    Returns:
        DataFrame with dropped columns.
    """
    return df.drop(columns=['id', 'CustomerId', 'Surname'])


def split_data(df: pd.DataFrame, target_col: str = 'Exited', test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and validation sets using stratification on the target column.
    
    Args:
        df: DataFrame to split.
        target_col: Name of the target column.
        test_size: Proportion of the dataset to include in the validation split.
        random_state: Random seed.
    
    Returns:
        Tuple of train and validation DataFrames.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[target_col])


def separate_features_and_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target from the dataset.
    
    Args:
        df: DataFrame containing features and target.
        target_col: Name of the target column.
    
    Returns:
        Tuple of (features, target).
    """
    input_cols = df.drop(columns=[target_col]).columns.tolist()
    return df[input_cols].copy(), df[target_col].copy()


def transform_balance_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transformation to the 'Balance' column.
    
    Args:
        df: DataFrame containing the 'Balance' column.
    
    Returns:
        Transformed DataFrame.
    """
    df['Balance'] = np.log1p(df['Balance'])
    return df


def scale_numeric_features(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scale numeric features using MinMaxScaler.
    
    Args:
        train_df: Training features DataFrame.
        val_df: Validation features DataFrame.
        numeric_cols: List of numeric column names.
    
    Returns:
        Tuple of scaled train_df, val_df, and the fitted scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])
    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    return train_df, val_df, scaler


def encode_categorical_features(train_df: pd.DataFrame, val_df: pd.DataFrame, categorical_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, list]:
    """
    Encode categorical features using OneHotEncoder.
    
    Args:
        train_df: Training features DataFrame.
        val_df: Validation features DataFrame.
        categorical_cols: List of categorical column names.
    
    Returns:
        Tuple of encoded train_df, val_df, fitted encoder, and list of encoded column names.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    
    train_encoded = encoder.transform(train_df[categorical_cols])
    val_encoded = encoder.transform(val_df[categorical_cols])
    
    train_df[encoded_cols] = train_encoded
    val_df[encoded_cols] = val_encoded
    
    return train_df, val_df, encoder, encoded_cols


def drop_original_categorical_columns(df: pd.DataFrame, cols_to_drop: list = ['Geography', 'Gender']) -> pd.DataFrame:
    """
    Drop original categorical columns after encoding.
    
    Args:
        df: DataFrame to modify.
        cols_to_drop: List of columns to drop.
    
    Returns:
        Modified DataFrame.
    """
    return df.drop(columns=cols_to_drop)


def preprocess_data(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Full preprocessing pipeline: drop unused columns, split, scale, encode, and prepare training/validation sets.
    
    Args:
        raw_df: Raw input DataFrame.
    
    Returns:
        Dictionary containing training/validation sets, scaler, encoder, and input column names.
    """
    df_cleaned = drop_unused_columns(raw_df)
    train_df, val_df = split_data(df_cleaned, target_col='Exited')

    train_inputs, train_targets = separate_features_and_target(train_df, 'Exited')
    val_inputs, val_targets = separate_features_and_target(val_df, 'Exited')

    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()

    train_inputs = transform_balance_column(train_inputs)
    val_inputs = transform_balance_column(val_inputs)

    train_inputs, val_inputs, scaler = scale_numeric_features(train_inputs, val_inputs, numeric_cols)
    train_inputs, val_inputs, encoder, encoded_cols = encode_categorical_features(train_inputs, val_inputs, categorical_cols)

    X_train = drop_original_categorical_columns(train_inputs)
    X_val = drop_original_categorical_columns(val_inputs)

    return {
        'X_train': X_train,
        'train_targets': train_targets,
        'X_val': X_val,
        'val_targets': val_targets,
        'input_cols': list(df_cleaned.columns[:-1]),
        'scaler': scaler,
        'encoder': encoder,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }


def preprocess_new_data(
    new_df: pd.DataFrame,
    scaler: MinMaxScaler,
    encoder: OneHotEncoder,
    numeric_cols: list,
    categorical_cols: list,
    cols_to_drop: list = ['Geography', 'Gender']
) -> pd.DataFrame:
    """
    Preprocess new (unseen) data using pre-fitted scaler and encoder.
    
    Steps:
    - Drop unused columns
    - Log-transform 'Balance'
    - Scale numeric features with provided scaler
    - Encode categorical features with provided encoder
    - Drop original categorical columns
    
    Args:
        new_df: Raw new data as a DataFrame.
        scaler: Pre-fitted MinMaxScaler.
        encoder: Pre-fitted OneHotEncoder.
        numeric_cols: List of numeric column names used during training.
        categorical_cols: List of categorical column names used during training.
        cols_to_drop: Categorical columns to drop after encoding.
    
    Returns:
        Preprocessed DataFrame ready for prediction.
    """
    df = drop_unused_columns(new_df.copy())
    df = transform_balance_column(df)

    # Scale numeric features
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Encode categorical features
    encoded_array = encoder.transform(df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    df[encoded_cols] = encoded_array

    # Drop original categorical columns
    df = drop_original_categorical_columns(df, cols_to_drop)

    return df
