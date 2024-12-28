import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Definicje kolumn
cols_to_scale = ["time", "age", "wtkg", "karnof", "preanti", "cd40", "cd80"]
cols_to_one_hot = ["trt", "strat"]


def _simplify(x: pd.DataFrame) -> pd.DataFrame:
    x.drop(columns={"z30", "treat", "str2", "cd420", "cd820"}, inplace=True)
    return x


# Funkcja do skalowania kolumn numerycznych
def _scale_numerical_columns(df: pd.DataFrame, cols_to_scale: list) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cols_to_scale])
    scaled_df = pd.DataFrame(scaled_data, columns=cols_to_scale, index=df.index)
    return scaled_df


# Funkcja do kodowania kolumn kategorycznych
def _encode_categorical_columns(df: pd.DataFrame, cols_to_one_hot) -> pd.DataFrame:
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded_data = encoder.fit_transform(df[cols_to_one_hot])
    encoded_cols = encoder.get_feature_names_out(cols_to_one_hot)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
    return encoded_df


# Funkcja do przetwarzania danych
def preprocess_aids(df: pd.DataFrame) -> pd.DataFrame:
    df = _simplify(df)

    # Skalowanie kolumn numerycznych
    scaled_df = _scale_numerical_columns(df, cols_to_scale)
    # Kodowanie kolumn kategorycznych
    encoded_df = _encode_categorical_columns(df, cols_to_one_hot)
    # Pobranie oryginalnych kolumn

    original_columns = [
        col for col in df.columns if col not in (cols_to_scale + cols_to_one_hot)
    ]
    original_df = df[original_columns].reset_index(drop=True)
    # Połączenie wynikowych danych
    final_df = pd.concat([original_df, scaled_df, encoded_df], axis=1)
    return final_df
