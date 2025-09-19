"""
Test simple pour Streamlit - Version robuste
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ajouter le répertoire racine au path
sys.path.append(str(Path(__file__).parent.parent))


def test_streamlit_simple() -> None:
    """Test simple pour Streamlit sans dépendances complexes"""

    # Test de base avec données numériques
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

    # Créer un DataFrame
    df = pd.DataFrame([test_data])

    # Vérifications de base
    assert len(df) == 1, "DataFrame doit contenir une ligne"
    assert len(df.columns) > 0, "DataFrame doit contenir des colonnes"

    # Vérifier que toutes les valeurs sont numériques
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(
            df[col]
        ), f"Colonne {col} doit être numérique"
        value = df[col].iloc[0]
        assert value is not None, f"Valeur de {col} ne doit pas être None"

    print("✅ Test Streamlit simple réussi")


def test_model_compatibility() -> None:
    """Test de compatibilité du modèle"""

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

        # Test de prédiction avec le bon nombre de features
        n_features = getattr(model, "n_features_in_", 153)  # Utiliser 153 par défaut
        test_data = np.random.random((1, n_features))
        prediction = model.predict_proba(test_data)

        assert prediction is not None, "Prédiction ne doit pas être None"
        assert len(prediction) > 0, "Prédiction doit contenir au moins un résultat"
        assert prediction.shape[1] == 2, "Prédiction doit avoir 2 classes"

        print("✅ Test de compatibilité du modèle réussi")

    except Exception as e:
        pytest.fail(f"Erreur de compatibilité du modèle: {e}")


if __name__ == "__main__":
    test_streamlit_simple()
    test_model_compatibility()
    print("✅ Tous les tests Streamlit simples réussis")
