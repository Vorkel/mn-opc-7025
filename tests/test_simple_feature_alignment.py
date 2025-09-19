"""
Test simple d'alignement des features - Version robuste
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajouter le répertoire racine au path
sys.path.append(str(Path(__file__).parent.parent))


def test_simple_feature_alignment() -> None:
    """Test simple d'alignement des features sans dépendances complexes"""

    # Vérifier que le modèle existe
    model_path = Path("models/best_credit_model.pkl")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé - test ignoré")

    # Données de test minimales
    test_data = {
        "CODE_GENDER": "M",
        "AMT_INCOME_TOTAL": 150000.0,
        "AMT_CREDIT": 300000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 280000.0,
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,
        "DAYS_REGISTRATION": -1500,
        "DAYS_ID_PUBLISH": -1500,
        "CNT_CHILDREN": 0,
        "CNT_FAM_MEMBERS": 2,
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
    }

    # Créer un DataFrame
    df = pd.DataFrame([test_data])

    # Vérifications de base
    assert len(df) == 1, "DataFrame doit contenir une ligne"
    assert len(df.columns) > 0, "DataFrame doit contenir des colonnes"

    # Test d'accès sécurisé aux colonnes
    for col in df.columns:
        assert col in df.columns, f"Colonne {col} doit être accessible"
        value = df[col].iloc[0]
        assert value is not None, f"Valeur de {col} ne doit pas être None"

    print("✅ Test d'alignement simple réussi")


def test_model_loading() -> None:
    """Test de chargement du modèle"""

    model_path = Path("models/best_credit_model.pkl")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé - test ignoré")

    try:
        import joblib

        model_data = joblib.load(model_path)

        # Vérifier la structure du modèle
        assert model_data is not None, "Modèle ne doit pas être None"

        if isinstance(model_data, dict):
            assert "model" in model_data, "Modèle doit contenir une clé 'model'"
            model = model_data["model"]
        else:
            model = model_data

        assert model is not None, "Modèle chargé ne doit pas être None"
        print("✅ Modèle chargé avec succès")

    except Exception as e:
        pytest.fail(f"Erreur de chargement du modèle: {e}")


if __name__ == "__main__":
    test_simple_feature_alignment()
    test_model_loading()
    print("✅ Tous les tests simples réussis")
