# complete_feature_engineering.py
# Pipeline complet de feature engineering pour générer les 153 features attendues

import pandas as pd
import numpy as np
from datetime import datetime


def create_complete_feature_set(client_data: dict) -> pd.DataFrame:
    """
    Génère les 153 features complètes attendues par le modèle à partir des
    données de base du client.

    Cette fonction prend les données de base du formulaire Streamlit et génère
    toutes les features nécessaires pour faire une prédiction avec le modèle
    entraîné.

    Args:
        client_data (dict): Données de base du client

    Returns:
        pd.DataFrame: DataFrame avec les 153 features dans l'ordre attendu
    """

    # Créer un DataFrame avec les données de base
    df = pd.DataFrame([client_data])

    print(f"🔧 Données de base reçues: {len(client_data)} features")

    # ========================================================================================
    # 1. FEATURES DE BASE (39 features) - À partir du formulaire Streamlit
    # ========================================================================================

    # Variables catégorielles (déjà présentes)
    base_categorical = [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
    ]

    # Variables numériques (déjà présentes)
    base_numeric = [
        "CNT_CHILDREN",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "REGION_POPULATION_RELATIVE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "OWN_CAR_AGE",
        "CNT_FAM_MEMBERS",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "HOUR_APPR_PROCESS_START",
    ]

    # Flags de contact (déjà présents)
    contact_flags = [
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
    ]

    # Variables de région (utiliser des valeurs par défaut)
    region_vars = [
        "WEEKDAY_APPR_PROCESS_START",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
    ]

    # Variables de cercle social
    social_vars = [
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "DAYS_LAST_PHONE_CHANGE",
    ]

    # Ajouter les variables manquantes avec des valeurs par défaut
    for var in region_vars:
        if var not in df.columns:
            if var == "WEEKDAY_APPR_PROCESS_START":
                df[var] = 2  # Mardi par défaut
            else:
                df[var] = 0  # Pas de différence par défaut

    for var in social_vars:
        if var not in df.columns:
            if var == "DAYS_LAST_PHONE_CHANGE":
                df[var] = -1000.0  # Valeur typique
            else:
                df[var] = 0.0  # Moyenne

    # ========================================================================================
    # 2. EXTERNAL SOURCES (3 features)
    # ========================================================================================

    ext_sources = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    for source in ext_sources:
        if source not in df.columns:
            # Utiliser des valeurs typiques basées sur les données d'entraînement
            df[source] = np.random.normal(0.5, 0.2)  # Valeurs entre 0 et 1

    # ========================================================================================
    # 3. FEATURES D'APPARTEMENT/BÂTIMENT (42 features)
    # ========================================================================================

    # Suffixes pour les statistiques de bâtiment
    building_suffixes = ["_AVG", "_MODE", "_MEDI"]

    # Types de données de bâtiment
    building_features = [
        "APARTMENTS",
        "BASEMENTAREA",
        "YEARS_BEGINEXPLUATATION",
        "YEARS_BUILD",
        "COMMONAREA",
        "ELEVATORS",
        "ENTRANCES",
        "FLOORSMAX",
        "FLOORSMIN",
        "LANDAREA",
        "LIVINGAPARTMENTS",
        "LIVINGAREA",
        "NONLIVINGAPARTMENTS",
        "NONLIVINGAREA",
    ]

    # Générer toutes les combinations
    for feature in building_features:
        for suffix in building_suffixes:
            col_name = feature + suffix
            if col_name not in df.columns:
                # Valeurs par défaut basées sur le type de feature
                if "YEARS" in feature:
                    df[col_name] = 20.0  # 20 ans en moyenne
                elif "AREA" in feature:
                    df[col_name] = 50.0  # 50 m² en moyenne
                elif feature in ["APARTMENTS", "ELEVATORS", "ENTRANCES", "FLOORS"]:
                    df[col_name] = 1.0  # 1 unité par défaut
                else:
                    df[col_name] = 0.0

    # Features spéciales de bâtiment
    building_special = [
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "TOTALAREA_MODE",
        "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE",
    ]

    for feature in building_special:
        if feature not in df.columns:
            df[feature] = 0  # Utiliser 0 au lieu de string

    # ========================================================================================
    # 4. FEATURES DE DOCUMENTS (20 features)
    # ========================================================================================

    document_flags = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]

    for flag in document_flags:
        if flag not in df.columns:
            # La plupart des documents ne sont pas fournis
            df[flag] = 0

    # ========================================================================================
    # 5. FEATURES DE BUREAU DE CRÉDIT (6 features)
    # ========================================================================================

    credit_bureau_features = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]

    for feature in credit_bureau_features:
        if feature not in df.columns:
            # Peu de demandes de crédit récentes par défaut
            if "HOUR" in feature or "DAY" in feature:
                df[feature] = 0.0
            elif "WEEK" in feature:
                df[feature] = 0.0
            elif "MON" in feature:
                df[feature] = 1.0  # 1 demande par mois
            elif "QRT" in feature:
                df[feature] = 2.0  # 2 demandes par trimestre
            elif "YEAR" in feature:
                df[feature] = 5.0  # 5 demandes par an

    # ========================================================================================
    # 6. FEATURES ENGINEERED (33 features)
    # ========================================================================================

    # Variables temporelles
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365.25
    df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"] / 365.25

    # Traitement des emplois anormaux (valeurs positives)
    df["DAYS_EMPLOYED_ABNORMAL"] = (df["DAYS_EMPLOYED"] > 0).astype(int)
    df.loc[df["DAYS_EMPLOYED"] > 0, "EMPLOYMENT_YEARS"] = 0

    # Autres variables temporelles
    df["YEARS_SINCE_REGISTRATION"] = (-df["DAYS_REGISTRATION"]) / 365.25
    df["YEARS_SINCE_ID_PUBLISH"] = (-df["DAYS_ID_PUBLISH"]) / 365.25

    # Groupes d'âge
    df["AGE_GROUP"] = pd.cut(
        df["AGE_YEARS"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=["Young", "Adult", "Middle", "Senior", "Elder"],
    ).astype(str)

    # Groupes d'emploi
    df["EMPLOYMENT_GROUP"] = pd.cut(
        df["EMPLOYMENT_YEARS"],
        bins=[-1, 0, 2, 5, 10, 50],
        labels=["Unemployed", "New", "Short", "Medium", "Long"],
    ).astype(str)

    # Ratio âge/emploi
    df["AGE_EMPLOYMENT_RATIO"] = df["AGE_YEARS"] / (df["EMPLOYMENT_YEARS"] + 1)

    # Ratios financiers
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)

    # Durée de crédit estimée
    df["CREDIT_DURATION"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)

    # Variables par personne
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    df["CREDIT_PER_PERSON"] = df["AMT_CREDIT"] / (df["CNT_FAM_MEMBERS"] + 1)

    # Groupes de revenus
    df["INCOME_GROUP"] = pd.cut(
        df["AMT_INCOME_TOTAL"],
        bins=[0, 100000, 200000, 500000, 1000000, np.inf],
        labels=["Low", "Medium", "High", "Very High", "Ultra High"],
    ).astype(str)

    # Groupes de crédit
    df["CREDIT_GROUP"] = pd.cut(
        df["AMT_CREDIT"],
        bins=[0, 200000, 500000, 1000000, 2000000, np.inf],
        labels=["Small", "Medium", "Large", "Very Large", "Ultra Large"],
    ).astype(str)

    # Indicateurs de propriété
    df["OWNS_PROPERTY"] = (
        (df["FLAG_OWN_CAR"] == "Y") & (df["FLAG_OWN_REALTY"] == "Y")
    ).astype(int)
    df["OWNS_NEITHER"] = (
        (df["FLAG_OWN_CAR"] == "N") & (df["FLAG_OWN_REALTY"] == "N")
    ).astype(int)

    # Scores agrégés
    contact_features = [
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
    ]
    df["CONTACT_SCORE"] = df[contact_features].sum(axis=1)

    # Score de documents
    doc_features = [col for col in df.columns if col.startswith("FLAG_DOCUMENT_")]
    df["DOCUMENT_SCORE"] = df[doc_features].sum(axis=1)

    # Score de région normalisé
    df["REGION_SCORE_NORMALIZED"] = 4 - df["REGION_RATING_CLIENT"]

    # Features externes
    external_features = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCES_MEAN"] = df[external_features].mean(axis=1)
    df["EXT_SOURCES_MAX"] = df[external_features].max(axis=1)
    df["EXT_SOURCES_MIN"] = df[external_features].min(axis=1)
    df["EXT_SOURCES_STD"] = df[external_features].std(axis=1)
    df["EXT_SOURCES_COUNT"] = df[external_features].count(axis=1)

    # Interaction âge et sources externes
    df["AGE_EXT_SOURCES_INTERACTION"] = df["AGE_YEARS"] * df["EXT_SOURCES_MEAN"]

    # ========================================================================================
    # 7. FEATURES MANQUANTES (5 features)
    # ========================================================================================

    missing_features = [
        "AMT_ANNUITY_MISSING",
        "AMT_GOODS_PRICE_MISSING",
        "DAYS_EMPLOYED_MISSING",
        "CNT_FAM_MEMBERS_MISSING",
        "DAYS_REGISTRATION_MISSING",
    ]

    # Indicateurs de valeurs manquantes
    df["AMT_ANNUITY_MISSING"] = df["AMT_ANNUITY"].isnull().astype(int)
    df["AMT_GOODS_PRICE_MISSING"] = df["AMT_GOODS_PRICE"].isnull().astype(int)
    df["DAYS_EMPLOYED_MISSING"] = df["DAYS_EMPLOYED"].isnull().astype(int)
    df["CNT_FAM_MEMBERS_MISSING"] = df["CNT_FAM_MEMBERS"].isnull().astype(int)
    df["DAYS_REGISTRATION_MISSING"] = df["DAYS_REGISTRATION"].isnull().astype(int)

    # ========================================================================================
    # 8. ENCODAGE DES VARIABLES CATÉGORIELLES
    # ========================================================================================

    # Mappings pour l'encodage des variables catégorielles
    encodings = {
        "NAME_CONTRACT_TYPE": {"Cash loans": 0, "Revolving loans": 1},
        "CODE_GENDER": {"M": 0, "F": 1, "XNA": 2},
        "FLAG_OWN_CAR": {"N": 0, "Y": 1},
        "FLAG_OWN_REALTY": {"N": 0, "Y": 1},
        "NAME_TYPE_SUITE": {
            "Unaccompanied": 0,
            "Family": 1,
            "Spouse, partner": 2,
            "Children": 3,
            "Other_A": 4,
            "Other_B": 5,
            "Group of people": 6,
        },
        "NAME_INCOME_TYPE": {
            "Working": 0,
            "State servant": 1,
            "Commercial associate": 2,
            "Pensioner": 3,
            "Unemployed": 4,
            "Student": 5,
            "Businessman": 6,
            "Maternity leave": 7,
        },
        "NAME_EDUCATION_TYPE": {
            "Secondary / secondary special": 0,
            "Higher education": 1,
            "Incomplete higher": 2,
            "Lower secondary": 3,
            "Academic degree": 4,
        },
        "NAME_FAMILY_STATUS": {
            "Single / not married": 0,
            "Married": 1,
            "Civil marriage": 2,
            "Widow": 3,
            "Separated": 4,
            "Unknown": 5,
        },
        "NAME_HOUSING_TYPE": {
            "House / apartment": 0,
            "Rented apartment": 1,
            "With parents": 2,
            "Municipal apartment": 3,
            "Office apartment": 4,
            "Co-op apartment": 5,
        },
        "AGE_GROUP": {"Young": 0, "Adult": 1, "Middle": 2, "Senior": 3, "Elder": 4},
        "EMPLOYMENT_GROUP": {
            "Unemployed": 0,
            "New": 1,
            "Short": 2,
            "Medium": 3,
            "Long": 4,
        },
        "INCOME_GROUP": {
            "Low": 0,
            "Medium": 1,
            "High": 2,
            "Very High": 3,
            "Ultra High": 4,
        },
        "CREDIT_GROUP": {
            "Small": 0,
            "Medium": 1,
            "Large": 2,
            "Very Large": 3,
            "Ultra Large": 4,
        },
    }

    # Appliquer les encodages
    for col, mapping in encodings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # Encoder OCCUPATION_TYPE et ORGANIZATION_TYPE (beaucoup de valeurs)
    for col in ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"]:
        if col in df.columns:
            # Utiliser un encodage numérique simple pour les strings
            if df[col].dtype == "object":
                unique_values = df[col].unique()
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                df[col] = df[col].map(mapping).fillna(0)

    # Encoder toutes les colonnes string restantes
    for col in df.columns:
        if df[col].dtype == "object":
            # Mapping par défaut pour les strings non gérées
            unique_values = df[col].unique()
            if len(unique_values) <= 50:  # Eviter les mappings trop grands
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                df[col] = df[col].map(mapping).fillna(0)
            else:
                # Pour les colonnes avec trop de valeurs uniques, utiliser hash
                df[col] = df[col].apply(
                    lambda x: hash(str(x)) % 1000 if x is not None else 0
                )

    # ========================================================================================
    # 9. ORDRE DES FEATURES ET FINALISATION
    # ========================================================================================

    # Liste complète des 153 features dans l'ordre attendu par le modèle
    expected_features = [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "CNT_CHILDREN",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "REGION_POPULATION_RELATIVE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "OWN_CAR_AGE",
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "OCCUPATION_TYPE",
        "CNT_FAM_MEMBERS",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "WEEKDAY_APPR_PROCESS_START",
        "HOUR_APPR_PROCESS_START",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
        "ORGANIZATION_TYPE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "APARTMENTS_AVG",
        "BASEMENTAREA_AVG",
        "YEARS_BEGINEXPLUATATION_AVG",
        "YEARS_BUILD_AVG",
        "COMMONAREA_AVG",
        "ELEVATORS_AVG",
        "ENTRANCES_AVG",
        "FLOORSMAX_AVG",
        "FLOORSMIN_AVG",
        "LANDAREA_AVG",
        "LIVINGAPARTMENTS_AVG",
        "LIVINGAREA_AVG",
        "NONLIVINGAPARTMENTS_AVG",
        "NONLIVINGAREA_AVG",
        "APARTMENTS_MODE",
        "BASEMENTAREA_MODE",
        "YEARS_BEGINEXPLUATATION_MODE",
        "YEARS_BUILD_MODE",
        "COMMONAREA_MODE",
        "ELEVATORS_MODE",
        "ENTRANCES_MODE",
        "FLOORSMAX_MODE",
        "FLOORSMIN_MODE",
        "LANDAREA_MODE",
        "LIVINGAPARTMENTS_MODE",
        "LIVINGAREA_MODE",
        "NONLIVINGAPARTMENTS_MODE",
        "NONLIVINGAREA_MODE",
        "APARTMENTS_MEDI",
        "BASEMENTAREA_MEDI",
        "YEARS_BEGINEXPLUATATION_MEDI",
        "YEARS_BUILD_MEDI",
        "COMMONAREA_MEDI",
        "ELEVATORS_MEDI",
        "ENTRANCES_MEDI",
        "FLOORSMAX_MEDI",
        "FLOORSMIN_MEDI",
        "LANDAREA_MEDI",
        "LIVINGAPARTMENTS_MEDI",
        "LIVINGAREA_MEDI",
        "NONLIVINGAPARTMENTS_MEDI",
        "NONLIVINGAREA_MEDI",
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "TOTALAREA_MODE",
        "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE",
        "DEF_60_CNT_SOCIAL_CIRCLE",
        "DAYS_LAST_PHONE_CHANGE",
        "FLAG_DOCUMENT_2",
        "FLAG_DOCUMENT_3",
        "FLAG_DOCUMENT_4",
        "FLAG_DOCUMENT_5",
        "FLAG_DOCUMENT_6",
        "FLAG_DOCUMENT_7",
        "FLAG_DOCUMENT_8",
        "FLAG_DOCUMENT_9",
        "FLAG_DOCUMENT_10",
        "FLAG_DOCUMENT_11",
        "FLAG_DOCUMENT_12",
        "FLAG_DOCUMENT_13",
        "FLAG_DOCUMENT_14",
        "FLAG_DOCUMENT_15",
        "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_17",
        "FLAG_DOCUMENT_18",
        "FLAG_DOCUMENT_19",
        "FLAG_DOCUMENT_20",
        "FLAG_DOCUMENT_21",
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "AGE_YEARS",
        "EMPLOYMENT_YEARS",
        "DAYS_EMPLOYED_ABNORMAL",
        "YEARS_SINCE_REGISTRATION",
        "YEARS_SINCE_ID_PUBLISH",
        "AGE_GROUP",
        "EMPLOYMENT_GROUP",
        "AGE_EMPLOYMENT_RATIO",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_GOODS_RATIO",
        "ANNUITY_CREDIT_RATIO",
        "CREDIT_DURATION",
        "INCOME_PER_PERSON",
        "CREDIT_PER_PERSON",
        "INCOME_GROUP",
        "CREDIT_GROUP",
        "OWNS_PROPERTY",
        "OWNS_NEITHER",
        "CONTACT_SCORE",
        "DOCUMENT_SCORE",
        "REGION_SCORE_NORMALIZED",
        "EXT_SOURCES_MEAN",
        "EXT_SOURCES_MAX",
        "EXT_SOURCES_MIN",
        "EXT_SOURCES_STD",
        "EXT_SOURCES_COUNT",
        "AGE_EXT_SOURCES_INTERACTION",
        "AMT_ANNUITY_MISSING",
        "AMT_GOODS_PRICE_MISSING",
        "DAYS_EMPLOYED_MISSING",
        "CNT_FAM_MEMBERS_MISSING",
        "DAYS_REGISTRATION_MISSING",
    ]

    # S'assurer que toutes les features sont présentes
    for feature in expected_features:
        if feature not in df.columns:
            print(f"⚠️ Feature manquante ajoutée: {feature}")
            df[feature] = 0.0

    # Réorganiser les colonnes dans l'ordre attendu
    df_final = df[expected_features].copy()

    # Remplacer les valeurs infinies et NaN
    df_final = df_final.replace([np.inf, -np.inf], 0)
    df_final = df_final.fillna(0)

    print(f"✅ Pipeline complet: {len(df_final.columns)} features générées")
    print(f"📊 Shape finale: {df_final.shape}")

    return df_final


def test_complete_pipeline():
    """
    Test du pipeline complet avec des données d'exemple
    """
    print("🧪 TEST DU PIPELINE COMPLET")
    print("=" * 40)

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

    # Tester le pipeline
    result = create_complete_feature_set(sample_data)

    print(f"📊 Features générées: {len(result.columns)}")
    print(f"📊 Shape: {result.shape}")

    # Vérifier qu'il n'y a pas de valeurs manquantes
    missing = result.isnull().sum().sum()
    print(f"📊 Valeurs manquantes: {missing}")

    return result


if __name__ == "__main__":
    # Test du pipeline
    result = test_complete_pipeline()

    # Sauvegarder le résultat pour inspection
    result.to_csv("test_complete_features.csv", index=False)
    print("💾 Résultat sauvegardé dans 'test_complete_features.csv'")
