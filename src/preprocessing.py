"""Preparação didática e sem vazamento de dados para o Titanic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
}

REQUIRED_FEATURE_COLUMNS = REQUIRED_COLUMNS.difference({"Survived"})

MODEL_SOURCE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
PCLASS_CATEGORIES = [1, 2, 3]
EMBARKED_CATEGORIES = ["C", "Q", "S"]


@dataclass(frozen=True)
class PreprocessingParameters:
    """Estatísticas ajustadas exclusivamente no conjunto de treinamento."""

    age_median: float
    embarked_mode: str
    fare_median: float


@dataclass(frozen=True)
class MinMaxParameters:
    """Mínimos e máximos de treino usados na normalização manual."""

    minimum: pd.Series
    maximum: pd.Series


def validate_titanic_dataset(data: pd.DataFrame, *, require_target: bool = True) -> None:
    """Confirma o schema do train.csv ou do test.csv oficial do Titanic."""
    required = REQUIRED_COLUMNS if require_target else REQUIRED_FEATURE_COLUMNS
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"CSV incompatível: colunas obrigatórias ausentes: {missing}")


def fit_preprocessor(train_data: pd.DataFrame) -> PreprocessingParameters:
    """Ajusta imputações no treino, evitando usar informações do teste."""
    validate_titanic_dataset(train_data)
    age_median = train_data["Age"].median()
    embarked_mode = train_data["Embarked"].mode(dropna=True)
    if pd.isna(age_median) or embarked_mode.empty:
        raise ValueError("Não foi possível calcular mediana de Age ou moda de Embarked no treino.")
    fare_median = train_data["Fare"].median()
    if pd.isna(fare_median):
        raise ValueError("Não foi possível calcular a mediana de Fare no treino.")
    return PreprocessingParameters(
        age_median=float(age_median),
        embarked_mode=str(embarked_mode.iloc[0]),
        fare_median=float(fare_median),
    )


def add_family_size(data: pd.DataFrame) -> pd.DataFrame:
    """Cria FamilySize: familiares a bordo mais o próprio passageiro."""
    result = data.copy()
    result["FamilySize"] = result["SibSp"] + result["Parch"] + 1
    return result


def transform_features(
    data: pd.DataFrame,
    parameters: PreprocessingParameters,
    *,
    use_family_size: bool = True,
) -> pd.DataFrame:
    """Imputa, codifica e retorna apenas features numéricas para o K-NN.

    PassengerId, Name, Ticket e Cabin não entram: não oferecem distância
    matemática significativa neste experimento introdutório.
    """
    # O test.csv externo não contém Survived, mas usa as mesmas features.
    validate_titanic_dataset(data, require_target=False)
    result = data.loc[:, MODEL_SOURCE_COLUMNS].copy()
    result["Age"] = result["Age"].fillna(parameters.age_median)
    result["Embarked"] = result["Embarked"].fillna(parameters.embarked_mode)
    # O train.csv oficial não possui Fare nulo; o test.csv externo possui um caso.
    result["Fare"] = result["Fare"].fillna(parameters.fare_median)

    selected_nulls = result.isna().sum()
    unresolved = selected_nulls[selected_nulls > 0]
    if not unresolved.empty:
        raise ValueError(f"Há NaN em colunas usadas pelo modelo: {unresolved.to_dict()}")

    result = add_family_size(result)
    result["Sex"] = result["Sex"].map({"male": 0, "female": 1})
    if result["Sex"].isna().any():
        raise ValueError("Sex contém categoria inesperada.")

    # Preservar o índice original é essencial após train_test_split, que não o reinicia.
    pclass = pd.Series(
        pd.Categorical(result.pop("Pclass"), categories=PCLASS_CATEGORIES),
        index=result.index,
    )
    embarked = pd.Series(
        pd.Categorical(result.pop("Embarked"), categories=EMBARKED_CATEGORIES),
        index=result.index,
    )
    if pclass.isna().any() or embarked.isna().any():
        raise ValueError("Pclass ou Embarked contém categoria inesperada.")

    pclass_dummies = pd.get_dummies(pclass, prefix="Pclass", dtype=int)
    # C é a categoria-base; Q e S são as duas colunas indicadoras restantes.
    embarked_dummies = pd.get_dummies(embarked, prefix="Embarked", drop_first=True, dtype=int)
    result = pd.concat([result, pclass_dummies, embarked_dummies], axis=1)

    if use_family_size:
        result = result.drop(columns=["SibSp", "Parch"])
    else:
        result = result.drop(columns=["FamilySize"])

    if result.isna().any().any():
        raise ValueError("A matriz final ainda contém NaN.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in result.dtypes):
        raise TypeError("A matriz final precisa conter apenas valores numéricos.")
    return result.astype(float)


def fit_min_max(train_features: pd.DataFrame, columns: Iterable[str]) -> MinMaxParameters:
    """Obtém min/max somente no treino para a fórmula manual de Min-Max."""
    columns = list(columns)
    return MinMaxParameters(
        minimum=train_features.loc[:, columns].min(),
        maximum=train_features.loc[:, columns].max(),
    )


def apply_min_max(
    features: pd.DataFrame,
    parameters: MinMaxParameters,
) -> pd.DataFrame:
    """Aplica x_norm = (x - x_min) / (x_max - x_min) com Pandas."""
    result = features.copy()
    columns = list(parameters.minimum.index)
    denominator = parameters.maximum - parameters.minimum
    non_constant = denominator[denominator != 0].index
    constant = denominator[denominator == 0].index
    result.loc[:, non_constant] = (
        result.loc[:, non_constant] - parameters.minimum.loc[non_constant]
    ) / denominator.loc[non_constant]
    # Uma feature constante não tem variação; representá-la por zero é seguro.
    if len(constant):
        result.loc[:, constant] = 0.0
    return result
