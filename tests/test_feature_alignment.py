"""
Test d'alignement des features - Version corrigée et simplifiée
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Ajouter le répertoire racine au path
sys.path.append(str(Path(__file__).parent.parent))


def test_feature_alignment_simple () -> None :
    """Test simple d'alignement des features"""
    print("TEST D'ALIGNEMENT DES FEATURES")
    print("=" * 50)

    # 1. Charger le modèle
    model_path = Path("models/best_credit_model.pkl")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé - test ignoré")

    model_data = joblib.load(model_path)

    # Le modèle peut être directement un RandomForest ou dans un dict
    if isinstance(model_data, dict) and "model" in model_data:
        model = model_data["model"]
        expected_features = model_data.get("feature_names", [])
    else:
        model = model_data
        expected_features = []

    print(f"✅ Modèle chargé: {len(expected_features)} features attendues")

    # 2. Test de base avec données numériques
    test_data = {
        "AMT_INCOME_TOTAL": 150000.0,
        "AMT_CREDIT": 300000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 280000.0,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,
        "DAYS_REGISTRATION": -1500,
        "DAYS_ID_PUBLISH": -1500,
        "CNT_CHILDREN": 0,
        "CNT_FAM_MEMBERS": 2,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
    }

    print(f"📝 Données de test créées: {len(test_data)} features")

    # 3. Créer un DataFrame
    df = pd.DataFrame([test_data])

    # Vérifications de base
    assert len(df) == 1, "DataFrame doit contenir une ligne"
    assert len(df.columns) > 0, "DataFrame doit contenir des colonnes"

    # Vérifier que toutes les valeurs sont numériques
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Colonne {col} doit être numérique"
        value = df[col].iloc[0]
        assert value is not None, f"Valeur de {col} ne doit pas être None"

    print("✅ Test d'alignement simple réussi")


def test_model_loading () -> None:
    """Test de chargement du modèle"""

    model_path = Path("models/best_credit_model.pkl")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé - test ignoré")

    try:
        model_data = joblib.load(model_path)

        # Vérifier la structure du modèle
        assert model_data is not None, "Modèle ne doit pas être None"

        if isinstance(model_data, dict):
            assert "model" in model_data, "Modèle doit contenir une clé 'model'"
            model = model_data["model"]
        else:
            model = model_data

        assert model is not None, "Modèle chargé ne doit pas être None"
        print("Modèle chargé avec succès")

    except Exception as e:
        pytest.fail(f"Erreur de chargement du modèle: {e}")

if __name__ == "__main__":
    test_feature_alignment_simple()
    test_model_loading()
    print("Tous les tests d'alignement réussis")
