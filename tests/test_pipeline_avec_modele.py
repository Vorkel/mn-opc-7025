# test_pipeline_avec_modele.py
# Test du pipeline complet avec le modèle

import sys
import pytest
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Ajouter le chemin vers src
sys.path.append(str(Path(__file__).parent.parent / "src"))
from src.feature_engineering import create_complete_feature_set


def test_avec_modele () -> None:
    """
    Test du pipeline complet avec prédiction du modèle
    """
    print("TEST PIPELINE + MODÈLE")
    print("=" * 40)

    # Charger le modèle
    try:
        model = joblib.load("models/best_credit_model.pkl")
        print(f"Type de modèle: {type(model)}")

        # Vérifier que c'est bien un RandomForest
        assert hasattr(model, 'predict_proba'), "Le modèle doit avoir la méthode predict_proba"

        # Données d'exemple du formulaire Streamlit
        sample_data = {
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": "M",
            "FLAG_OWN_CAR": "Y",
            "FLAG_OWN_REALTY": "Y",
            "CNT_CHILDREN": 2,
            "AMT_INCOME_TOTAL": 150000.0,
            "AMT_CREDIT": 500000.0,
            "AMT_ANNUITY": 25000.0,
            "AMT_GOODS_PRICE": 480000.0,
            "NAME_TYPE_SUITE": "Family",
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Higher education",
            "NAME_FAMILY_STATUS": "Married",
            "NAME_HOUSING_TYPE": "House / apartment",
            "REGION_POPULATION_RELATIVE": 0.035,
            "DAYS_BIRTH": -12000,
            "DAYS_EMPLOYED": -3000,
            "DAYS_REGISTRATION": -5000,
            "DAYS_ID_PUBLISH": -2000,
            "OWN_CAR_AGE": 5.0,
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": 1,
            "FLAG_WORK_PHONE": 0,
            "FLAG_CONT_MOBILE": 1,
            "FLAG_PHONE": 1,
            "FLAG_EMAIL": 1,
            "OCCUPATION_TYPE": "Laborers",
            "CNT_FAM_MEMBERS": 4.0,
            "REGION_RATING_CLIENT": 2,
            "REGION_RATING_CLIENT_W_CITY": 2,
            "HOUR_APPR_PROCESS_START": 14,
            "ORGANIZATION_TYPE": "Business Entity Type 3",
        }

        print("\nGÉNÉRATION DES FEATURES")
        # Générer les features avec le feature engineer
        df_features = create_complete_feature_set(sample_data)
        print(f"Features générées: {df_features.shape}")

        print("\nPRÉDICTION")
        # Faire la prédiction
        try:
            # Prédiction de probabilité
            proba = model.predict_proba(df_features)[0]
            prediction = model.predict(df_features)[0]

            print(f"Prédiction: {prediction}")
            print(f"Probabilité de défaut: {proba[1]:.4f}")
            print(f"Probabilité de non-défaut: {proba[0]:.4f}")

            assert True, "Test réussi"

        except Exception as e:
            print(f"Erreur prédiction: {e}")
            print(f"Shape des features: {df_features.shape}")
            print(f"Features attendues: {model.n_features_in_}")
            print(f"Colonnes avec NaN: {df_features.isnull().sum().sum()}")
            print(f"Dtypes: {df_features.dtypes.value_counts()}")

            # Afficher les premières valeurs pour debug
            print("\nPremières valeurs:")
            print(df_features.iloc[0, :10].to_dict())

            pytest.fail("Erreur dans le pipeline de test")

    except Exception as e:
        print(f"Erreur chargement modèle: {e}")
        pytest.fail(f"Erreur chargement modèle: {e}")


if __name__ == "__main__":
    test_avec_modele()
    print("\nSUCCÈS - Pipeline complet fonctionnel !")
