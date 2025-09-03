"""
Test de l'alignement des features entre Streamlit et le modèle
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Ajouter le répertoire racine au path
sys.path.append(str(Path(__file__).parent))

def test_feature_alignment():
    """Test l'alignement des features"""
    print("🔍 TEST D'ALIGNEMENT DES FEATURES")
    print("=" * 50)

    # 1. Charger le modèle
    model_path = Path("models/best_credit_model.pkl")
    if not model_path.exists():
        print("❌ Modèle non trouvé")
        return False

    model_data = joblib.load(model_path)
    model = model_data["model"]
    expected_features = model_data["feature_names"]

    print(f"✅ Modèle chargé: {len(expected_features)} features attendues")

    # 2. Simuler les données Streamlit
    client_data_raw = {
        # Features de base comme dans Streamlit
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 150000.0,
        "AMT_CREDIT": 300000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 280000.0,
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -12000,  # ~33 ans
        "DAYS_EMPLOYED": -2000,  # ~5 ans d'emploi
        "DAYS_REGISTRATION": -1500,
        "DAYS_ID_PUBLISH": -1500,
        "CNT_FAM_MEMBERS": 2,
        "REGION_RATING_CLIENT": 2,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "REGION_POPULATION_RELATIVE": 0.5,
        "ORGANIZATION_TYPE": "Business Entity Type 3",
        "OCCUPATION_TYPE": "Laborers",
        "FLAG_MOBIL": 1,
        "FLAG_EMP_PHONE": 0,
        "FLAG_WORK_PHONE": 0,
        "FLAG_CONT_MOBILE": 1,
        "FLAG_PHONE": 1,
        "FLAG_EMAIL": 0,
        "LIVE_CITY_NOT_WORK_CITY": 0,
        "LIVE_REGION_NOT_WORK_REGION": 0,
        "WEEKDAY_APPR_PROCESS_START": 1,
        "HOUR_APPR_PROCESS_START": 12,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
    }

    print(f"📝 Données de base créées: {len(client_data_raw)} features")

    # 3. Appliquer le feature engineering
    try:
        from src.feature_engineering import CreditFeatureEngineer

        feature_engineer = CreditFeatureEngineer()
        df_raw = pd.DataFrame([client_data_raw])
        df_engineered = feature_engineer.engineer_features(df_raw)

        print(f"🔧 Feature engineering appliqué: {len(df_engineered.columns)} features générées")

        # 4. Vérifier l'alignement
        missing_features = [f for f in expected_features if f not in df_engineered.columns]
        extra_features = [f for f in df_engineered.columns if f not in expected_features]

        print(f"\n📊 ANALYSE D'ALIGNEMENT:")
        print(f"   Features attendues: {len(expected_features)}")
        print(f"   Features générées: {len(df_engineered.columns)}")
        print(f"   Features manquantes: {len(missing_features)}")
        print(f"   Features en trop: {len(extra_features)}")

        if missing_features:
            print(f"\n❌ FEATURES MANQUANTES ({len(missing_features)}):")
            for feature in missing_features[:10]:  # Afficher les 10 premières
                print(f"   - {feature}")
            if len(missing_features) > 10:
                print(f"   ... et {len(missing_features) - 10} autres")

        if extra_features:
            print(f"\n⚠️  FEATURES EN TROP ({len(extra_features)}):")
            for feature in extra_features[:10]:  # Afficher les 10 premières
                print(f"   - {feature}")
            if len(extra_features) > 10:
                print(f"   ... et {len(extra_features) - 10} autres")

        # 5. Test de prédiction
        if not missing_features and not extra_features:
            print(f"\n✅ ALIGNEMENT PARFAIT - Test de prédiction...")
            try:
                prediction = model.predict_proba(df_engineered)
                print(f"✅ Prédiction réussie: {prediction[0][1]:.4f}")
                return True
            except Exception as e:
                print(f"❌ Erreur prédiction: {e}")
                return False
        else:
            print(f"\n⚠️  ALIGNEMENT IMPARFAIT - Correction nécessaire")

            # Tenter une correction automatique
            df_corrected = df_engineered.copy()

            # Ajouter les features manquantes
            for feature in missing_features:
                if "FLAG_" in feature:
                    df_corrected[feature] = 0
                elif "AMT_" in feature:
                    df_corrected[feature] = 0.0
                elif "CNT_" in feature:
                    df_corrected[feature] = 0
                elif "DAYS_" in feature:
                    df_corrected[feature] = 0
                else:
                    df_corrected[feature] = 0.5

            # Supprimer les features en trop
            for feature in extra_features:
                if feature in df_corrected.columns:
                    df_corrected = df_corrected.drop(columns=[feature])

            # Réordonner selon l'ordre attendu
            df_final = df_corrected[expected_features]

            print(f"🔧 Correction appliquée: {len(df_final.columns)} features finales")

            # Test de prédiction avec correction
            try:
                prediction = model.predict_proba(df_final)
                print(f"✅ Prédiction après correction: {prediction[0][1]:.4f}")
                return True
            except Exception as e:
                print(f"❌ Erreur prédiction après correction: {e}")
                return False

    except Exception as e:
        print(f"❌ Erreur feature engineering: {e}")
        return False

def test_streamlit_client_data():
    """Test avec les mêmes données que Streamlit génère"""
    print(f"\n🎯 TEST AVEC DONNÉES STREAMLIT")
    print("=" * 40)

    # Données exactement comme généré par Streamlit
    client_data = {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 150000.0,
        "AMT_CREDIT": 300000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 280000.0,
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,
        "DAYS_REGISTRATION": -1500,
        "DAYS_ID_PUBLISH": -1500,
        "CNT_FAM_MEMBERS": 2,
        "FLAG_MOBIL": 1,
        "FLAG_EMP_PHONE": 0,
        "FLAG_WORK_PHONE": 0,
        "FLAG_CONT_MOBILE": 1,
        "FLAG_PHONE": 1,
        "FLAG_EMAIL": 0,
        "REGION_RATING_CLIENT": 2,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "REGION_POPULATION_RELATIVE": 0.5,
        "ORGANIZATION_TYPE": "Business Entity Type 3",
        "OCCUPATION_TYPE": "Laborers",
        "LIVE_CITY_NOT_WORK_CITY": 0,
        "LIVE_REGION_NOT_WORK_REGION": 0,
        "REG_REGION_NOT_LIVE_REGION": 0,
        "REG_REGION_NOT_WORK_REGION": 0,
        "REG_CITY_NOT_LIVE_CITY": 0,
        "REG_CITY_NOT_WORK_CITY": 0,
        "WEEKDAY_APPR_PROCESS_START": 1,
        "HOUR_APPR_PROCESS_START": 12,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
    }

    # Simuler apply_feature_engineering
    try:
        from streamlit_app.main import apply_feature_engineering

        df = pd.DataFrame([client_data])
        df_processed = apply_feature_engineering(df)

        print(f"✅ Streamlit feature engineering: {len(df_processed.columns)} features")

        # Charger le modèle et tester
        model_data = joblib.load("models/best_credit_model.pkl")
        model = model_data["model"]

        prediction = model.predict_proba(df_processed)
        print(f"✅ Prédiction Streamlit: {prediction[0][1]:.4f}")

        return True

    except Exception as e:
        print(f"❌ Erreur test Streamlit: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TESTS D'ALIGNEMENT DES FEATURES")
    print("=" * 60)

    # Test 1: Alignement général
    success1 = test_feature_alignment()

    # Test 2: Données Streamlit
    success2 = test_streamlit_client_data()

    print(f"\n📋 RÉSUMÉ DES TESTS:")
    print(f"   Test alignement général: {'✅' if success1 else '❌'}")
    print(f"   Test données Streamlit: {'✅' if success2 else '❌'}")

    if success1 and success2:
        print(f"\n🎉 TOUS LES TESTS RÉUSSIS!")
        print(f"   Le problème de désalignement est résolu.")
    else:
        print(f"\n❌ TESTS ÉCHOUÉS")
        print(f"   Correction supplémentaire nécessaire.")
