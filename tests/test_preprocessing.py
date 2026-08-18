import pandas as pd
import pytest

from src.preprocessing import (
    EMBARKED_CATEGORIES,
    PCLASS_CATEGORIES,
    add_family_size,
    apply_min_max,
    fit_min_max,
    fit_preprocessor,
    transform_features,
    validate_titanic_dataset,
)


def titanic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived": [0, 1, 1, 0],
            "Pclass": [3, 1, 2, 3],
            "Name": ["A", "B", "C", "D"],
            "Sex": ["male", "female", "female", "male"],
            "Age": [22.0, None, 30.0, 40.0],
            "SibSp": [1, 1, 0, 2],
            "Parch": [0, 0, 1, 1],
            "Ticket": ["A/5", "PC", "STON", "X"],
            "Fare": [7.25, 71.28, 13.0, 8.0],
            "Cabin": [None, "C85", None, None],
            "Embarked": ["S", "C", None, "Q"],
        }
    )


def test_add_family_size_includes_passenger() -> None:
    result = add_family_size(titanic_frame())
    assert result["FamilySize"].tolist() == [2, 2, 2, 4]


def test_transformed_features_have_no_nan_and_are_numeric() -> None:
    data = titanic_frame()
    data.index = [10, 20, 30, 40]  # reproduz índices preservados pelo train_test_split
    features = transform_features(data, fit_preprocessor(data))
    assert not features.isna().any().any()
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes)
    assert features.index.tolist() == [10, 20, 30, 40]
    assert {"Pclass_1", "Pclass_2", "Pclass_3", "Embarked_Q", "Embarked_S"}.issubset(features.columns)


def test_sex_mapping_and_family_size_variant() -> None:
    data = titanic_frame()
    features = transform_features(data, fit_preprocessor(data), use_family_size=False)
    assert features["Sex"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert "FamilySize" not in features.columns
    assert {"SibSp", "Parch"}.issubset(features.columns)


def test_manual_min_max_uses_training_parameters() -> None:
    train = pd.DataFrame({"Age": [10.0, 20.0], "Fare": [5.0, 15.0]})
    test = pd.DataFrame({"Age": [15.0, 30.0], "Fare": [10.0, 20.0]})
    scaled_train = apply_min_max(train, fit_min_max(train, ["Age", "Fare"]))
    scaled_test = apply_min_max(test, fit_min_max(train, ["Age", "Fare"]))
    assert scaled_train.iloc[0].tolist() == [0.0, 0.0]
    assert scaled_train.iloc[1].tolist() == [1.0, 1.0]
    assert scaled_test.iloc[1].tolist() == [2.0, 1.5]


def test_required_columns_are_checked() -> None:
    with pytest.raises(ValueError, match="colunas obrigatórias"):
        validate_titanic_dataset(titanic_frame().drop(columns=["Cabin"]))


def test_expected_categories_are_documented() -> None:
    assert PCLASS_CATEGORIES == [1, 2, 3]
    assert EMBARKED_CATEGORIES == ["C", "Q", "S"]
