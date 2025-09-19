# test_streamlit_integration.py
# Test final de l'intégration Streamlit avec le nouveau pipeline

import sys
import pytest
from pathlib import Path
import pandas as pd
import joblib

# Ajouter le chemin pour les imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.feature_engineering import create_complete_feature_set


def test_streamlit_integration() -> None:
    """
    Test complet de l'intégration Streamlit avec données de formulaire
    """
    print("TEST INTÉGRATION STREAMLIT")
    print("=" * 50)

    # Charger le modèle
    try:
        model = joblib.load("models/best_credit_model.pkl")
        assert hasattr(
            model, "predict_proba"
        ), "Le modèle doit avoir la méthode predict_proba"
        print("Modèle chargé")
    except Exception as e:
        print(f"Erreur chargement modèle: {e}")
        assert False, f"Erreur chargement modèle: {e}"

    # Données simulées du formulaire Streamlit
    client_data_examples = [
        {
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
        },
        {
            "NAME_CONTRACT_TYPE": "Cash loans",
            "CODE_GENDER": "F",
            "FLAG_OWN_CAR": "N",
            "FLAG_OWN_REALTY": "N",
            "CNT_CHILDREN": 0,
            "AMT_INCOME_TOTAL": 80000.0,
            "AMT_CREDIT": 200000.0,
            "AMT_ANNUITY": 12000.0,
            "AMT_GOODS_PRICE": 180000.0,
            "NAME_TYPE_SUITE": "Unaccompanied",
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            "NAME_FAMILY_STATUS": "Single / not married",
            "NAME_HOUSING_TYPE": "Rented apartment",
            "REGION_POPULATION_RELATIVE": 0.02,
            "DAYS_BIRTH": -9000,
            "DAYS_EMPLOYED": -1500,
            "DAYS_REGISTRATION": -3000,
            "DAYS_ID_PUBLISH": -1500,
            "OWN_CAR_AGE": None,
            "FLAG_MOBIL": 1,
            "FLAG_EMP_PHONE": 0,
            "FLAG_WORK_PHONE": 1,
            "FLAG_CONT_MOBILE": 1,
            "FLAG_PHONE": 1,
            "FLAG_EMAIL": 0,
            "OCCUPATION_TYPE": "Sales staff",
            "CNT_FAM_MEMBERS": 1.0,
            "REGION_RATING_CLIENT": 3,
            "REGION_RATING_CLIENT_W_CITY": 3,
            "HOUR_APPR_PROCESS_START": 10,
            "ORGANIZATION_TYPE": "Business Entity Type 1",
        },
    ]

    success_count = 0

    for i, client_data in enumerate(client_data_examples, 1):
        print(f"\nTEST CLIENT {i}")
        print("-" * 30)

        try:
            # Générer les features avec notre pipeline
            df_engineered = create_complete_feature_set(client_data)
            print(f"Features générées: {df_engineered.shape}")

            # Faire une prédiction
            probabilities = model.predict_proba(df_engineered)
            probability = probabilities[0][1]
            prediction = model.predict(df_engineered)[0]

            # Déterminer la décision
            threshold = 0.5
            decision = "ACCORDÉ" if probability < threshold else "REFUSÉ"

            # Niveau de risque
            if probability < 0.3:
                risk_level = "Faible"
            elif probability < 0.7:
                risk_level = "Modéré"
            else:
                risk_level = "Élevé"

            print(f"Prédiction: {prediction}")
            print(f"Probabilité de défaut: {probability:.4f}")
            print(f"Décision: {decision}")
            print(f"Niveau de risque: {risk_level}")

            success_count += 1

        except Exception as e:
            print(f"Erreur client {i}: {e}")
            print(f"Type erreur: {type(e)}")
            continue

    print("\nRÉSUMÉ FINAL")
    print(f"✅ Clients testés avec succès: {success_count}/{len(client_data_examples)}")

    if success_count == len(client_data_examples):
        print("TOUS LES TESTS RÉUSSIS !")
        print("L'intégration Streamlit est fonctionnelle")
        assert True, "Tous les tests d'intégration Streamlit réussis"
    else:
        print("❌ Certains tests ont échoué")
        pytest.fail("Certains tests d'intégration Streamlit ont échoué")


if __name__ == "__main__":
    test_streamlit_integration()
    print("\nDÉPLOIEMENT RECOMMANDÉ")
    print("L'application peut être déployée sur Streamlit Cloud")
