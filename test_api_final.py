#!/usr/bin/env python3
"""
Test final de l'API avec notre pipeline corrigé
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

# Ajouter le chemin du projet
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.feature_engineering import create_complete_feature_set

def test_local_pipeline():
    """Test du pipeline en local"""
    print("🔧 TEST DU PIPELINE LOCAL")
    print("=" * 50)

    # Données d'exemple (format Streamlit complet)
    client_data = {
        "AMT_INCOME_TOTAL": 150000,
        "AMT_CREDIT": 600000,
        "AMT_ANNUITY": 25000,
        "AMT_GOODS_PRICE": 600000,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 1,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -3000,
        "DAYS_REGISTRATION": -1000,
        "DAYS_ID_PUBLISH": -2000,
        "CNT_FAM_MEMBERS": 3,
        "REGION_RATING_CLIENT": 2,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "FLAG_MOBIL": 1,
        "FLAG_EMP_PHONE": 1,
        "FLAG_WORK_PHONE": 0,
        "FLAG_CONT_MOBILE": 1,
        "FLAG_PHONE": 0,
        "FLAG_EMAIL": 1,
        "ORGANIZATION_TYPE": "Business Entity Type 3",
        "OCCUPATION_TYPE": "Managers",
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
        "HOUR_APPR_PROCESS_START": 14,
        "REG_REGION_NOT_LIVE_REGION": 0,
        "REG_REGION_NOT_WORK_REGION": 0,
        "LIVE_REGION_NOT_WORK_REGION": 0,
        "REG_CITY_NOT_LIVE_CITY": 0,
        "REG_CITY_NOT_WORK_CITY": 0,
        "LIVE_CITY_NOT_WORK_CITY": 0
    }

    print(f"📥 Données d'entrée: {len(client_data)} champs")

    # Test du feature engineering
    try:
        df_features = create_complete_feature_set(client_data)
        print(f"✅ Feature engineering réussi: {len(df_features.columns)} features générées")
        print(f"📊 Colonnes générées: {list(df_features.columns[:10])}... (10 premiers)")

        # Test avec le modèle
        model_path = project_root / "models" / "best_credit_model.pkl"
        if model_path.exists():
            print(f"\n🤖 Chargement du modèle: {model_path}")
            model_data = joblib.load(model_path)
            model = model_data['model']  # Le vrai modèle est dans le dict

            print(f"📋 Modèle: {type(model)}")
            print(f"🔑 Features attendues: {len(model_data['feature_names'])}")

            # Prédiction
            prediction = model.predict(df_features)[0]
            probability = model.predict_proba(df_features)[0]

            print(f"✅ Prédiction réussie!")
            print(f"📈 Classe prédite: {prediction}")
            print(f"📊 Probabilités: {probability}")
            print(f"🎯 Probabilité de défaut: {probability[1]:.3f}")

            if probability[1] < 0.5:
                print("🟢 CRÉDIT ACCEPTÉ")
            else:
                print("🔴 CRÉDIT REFUSÉ")

        else:
            print(f"⚠️ Modèle non trouvé: {model_path}")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_local_pipeline()
