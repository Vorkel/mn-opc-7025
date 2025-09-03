"""
Module de feature engineering pour le projet de scoring crédit
Basé sur le dataset Home Credit standard
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

# Configuration du logging
logger = logging.getLogger(__name__)

class CreditFeatureEngineer:
    """
    Classe pour l'ingénierie des features de scoring crédit
    Utilise uniquement les features standard du dataset Home Credit
    """

    def __init__(self) -> None:
        """Initialise le feature engineer avec les features standard"""
        # Features standard Home Credit (sans features fictives)
        self.training_features: List[str] = [
            # Informations personnelles
            "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
            "CNT_CHILDREN", "CNT_FAM_MEMBERS",

            # Éducation et statut
            "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",

            # Variables temporelles
            "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",

            # Revenus et finances
            "NAME_INCOME_TYPE", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
            "AMT_CREDIT", "AMT_GOODS_PRICE",

            # Type de suite et organisation
            "NAME_TYPE_SUITE", "ORGANIZATION_TYPE",

            # Features de base standard
            "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
            "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_8",

            # Features contextuelles
            "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
            "REGION_POPULATION_RELATIVE", "LIVE_CITY_NOT_WORK_CITY",
            "LIVE_REGION_NOT_WORK_REGION",

            # Features de contrat
            "NAME_CONTRACT_TYPE", "OCCUPATION_TYPE",

            # Features numériques de base
            "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG", "OWN_CAR_AGE",
            "FLOORSMIN_AVG", "FLOORSMIN_MEDI", "FLOORSMIN_MODE",
            "FONDKAPREMONT_MODE", "HOUR_APPR_PROCESS_START"
        ]

        # Features à supprimer si présentes (features fictives)
        self.features_to_remove: List[str] = [
            "DEBT_RATIO", "DISPOSABLE_INCOME", "BANK_YEARS",
            "PAYMENT_HISTORY", "OVERDRAFT_FREQUENCY", "SAVINGS_AMOUNT",
            "CREDIT_BUREAU_SCORE", "CREDIT_DURATION", "CREDIT_PURPOSE",
            "PERSONAL_CONTRIBUTION", "GUARANTEE_TYPE", "SPENDING_HABITS",
            "INCOME_STABILITY", "BALANCE_EVOLUTION", "SECTOR_ACTIVITY",
            "UNEMPLOYMENT_RATE", "REAL_ESTATE_TREND", "FLAG_DOCUMENT_1",
            "FONDKAPITAL_MODE", "REAL_ESTATE_TREND", "SECTOR_ACTIVITY"
        ]

        logger.info(f"Feature engineer initialisé avec {len(self.training_features)} features standard")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique l'ingénierie des features complète pour générer les 153 features

        Args:
            df: DataFrame avec les données brutes

        Returns:
            DataFrame avec toutes les features engineered (153 features)
        """
        try:
            logger.info("Début de l'ingénierie complète des features")
            df_engineered = df.copy()

            # 1. Supprimer les features fictives si présentes
            for feature in self.features_to_remove:
                if feature in df_engineered.columns:
                    df_engineered = df_engineered.drop(columns=[feature])
                    logger.info(f"Feature fictive supprimée: {feature}")

            # 2. Ajouter toutes les features manquantes avec des valeurs par défaut
            expected_features = [
                'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE',
                'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
                'OWN_CAR_AGE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
                'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE',
                'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
                'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
                'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
                'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE',
                'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
            ]

            # Ajouter features d'appartement
            building_prefixes = ['APARTMENTS', 'BASEMENTAREA', 'YEARS_BEGINEXPLUATATION',
                               'YEARS_BUILD', 'COMMONAREA', 'ELEVATORS', 'ENTRANCES',
                               'FLOORSMAX', 'FLOORSMIN', 'LANDAREA', 'LIVINGAPARTMENTS',
                               'LIVINGAREA', 'NONLIVINGAPARTMENTS', 'NONLIVINGAREA']
            building_suffixes = ['_AVG', '_MODE', '_MEDI']

            for prefix in building_prefixes:
                for suffix in building_suffixes:
                    expected_features.append(prefix + suffix)

            # Ajouter features spéciales de bâtiment
            expected_features.extend([
                'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE',
                'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
            ])

            # Ajouter features sociales
            expected_features.extend([
                'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
                'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE'
            ])

            # Ajouter features de documents
            for i in range(2, 22):
                expected_features.append(f'FLAG_DOCUMENT_{i}')

            # Ajouter features de bureau de crédit
            expected_features.extend([
                'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'
            ])

            # S'assurer que toutes les features de base sont présentes
            for feature in expected_features:
                if feature not in df_engineered.columns:
                    # Valeurs par défaut selon le type de feature
                    if "FLAG_" in feature:
                        df_engineered[feature] = 0
                    elif "EXT_SOURCE" in feature:
                        df_engineered[feature] = 0.5  # Valeur neutre
                    elif "_AVG" in feature or "_MODE" in feature or "_MEDI" in feature:
                        if "YEARS" in feature:
                            df_engineered[feature] = 20.0
                        elif "AREA" in feature:
                            df_engineered[feature] = 50.0
                        else:
                            df_engineered[feature] = 1.0
                    elif "AMT_REQ_CREDIT_BUREAU" in feature:
                        if "HOUR" in feature or "DAY" in feature:
                            df_engineered[feature] = 0.0
                        else:
                            df_engineered[feature] = 1.0
                    elif "CNT_" in feature:
                        df_engineered[feature] = 0.0
                    elif "DAYS_" in feature:
                        df_engineered[feature] = -1000.0
                    elif "REGION" in feature or "RATING" in feature:
                        df_engineered[feature] = 2
                    else:
                        df_engineered[feature] = 0

            # 3. Encodage des variables catégorielles standard
            categorical_mappings = {
                "CODE_GENDER": {"M": 1, "F": 0, "Homme": 1, "Femme": 0},
                "FLAG_OWN_CAR": {"Y": 1, "N": 0, "Oui": 1, "Non": 0},
                "FLAG_OWN_REALTY": {"Y": 1, "N": 0, "Oui": 1, "Non": 0},
                "NAME_CONTRACT_TYPE": {"Cash loans": 0, "Revolving loans": 1},
                "NAME_TYPE_SUITE": {
                    "Unaccompanied": 0, "Family": 1, "Spouse, partner": 2,
                    "Children": 3, "Other_A": 4, "Other_B": 5, "Group of people": 6
                },
                "NAME_INCOME_TYPE": {
                    "Working": 0, "State servant": 1, "Commercial associate": 2,
                    "Pensioner": 3, "Unemployed": 4, "Student": 5, "Businessman": 6
                },
                "NAME_EDUCATION_TYPE": {
                    "Secondary / secondary special": 0, "Higher education": 1,
                    "Incomplete higher": 2, "Lower secondary": 3, "Academic degree": 4
                },
                "NAME_FAMILY_STATUS": {
                    "Single / not married": 0, "Married": 1, "Civil marriage": 2,
                    "Widow": 3, "Separated": 4, "Unknown": 5
                },
                "NAME_HOUSING_TYPE": {
                    "House / apartment": 0, "Rented apartment": 1, "With parents": 2,
                    "Municipal apartment": 3, "Office apartment": 4, "Co-op apartment": 5
                }
            }

            # Appliquer les mappings
            for col, mapping in categorical_mappings.items():
                if col in df_engineered.columns:
                    df_engineered[col] = df_engineered[col].replace(mapping).fillna(0)

            # Encoder OCCUPATION_TYPE et ORGANIZATION_TYPE
            for col in ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']:
                if col in df_engineered.columns and df_engineered[col].dtype == 'object':
                    unique_values = df_engineered[col].unique()
                    mapping = {val: idx for idx, val in enumerate(unique_values)}
                    df_engineered[col] = df_engineered[col].map(mapping).fillna(0)

            # 4. Feature Engineering Avancé
            # Variables temporelles
            if "DAYS_BIRTH" in df_engineered.columns:
                df_engineered["AGE_YEARS"] = (-df_engineered["DAYS_BIRTH"]) / 365.25

            if "DAYS_EMPLOYED" in df_engineered.columns:
                df_engineered["EMPLOYMENT_YEARS"] = df_engineered["DAYS_EMPLOYED"] / 365.25
                df_engineered["DAYS_EMPLOYED_ABNORMAL"] = (df_engineered["DAYS_EMPLOYED"] > 0).astype(int)
                df_engineered.loc[df_engineered["DAYS_EMPLOYED"] > 0, "EMPLOYMENT_YEARS"] = 0

            if "DAYS_REGISTRATION" in df_engineered.columns:
                df_engineered["YEARS_SINCE_REGISTRATION"] = (-df_engineered["DAYS_REGISTRATION"]) / 365.25

            if "DAYS_ID_PUBLISH" in df_engineered.columns:
                df_engineered["YEARS_SINCE_ID_PUBLISH"] = (-df_engineered["DAYS_ID_PUBLISH"]) / 365.25

            # Groupes et ratios
            if "AGE_YEARS" in df_engineered.columns:
                df_engineered['AGE_GROUP'] = pd.cut(
                    df_engineered['AGE_YEARS'],
                    bins=[0, 25, 35, 50, 65, 100],
                    labels=[0, 1, 2, 3, 4]
                ).astype(float).fillna(1)

            if "EMPLOYMENT_YEARS" in df_engineered.columns:
                df_engineered['EMPLOYMENT_GROUP'] = pd.cut(
                    df_engineered['EMPLOYMENT_YEARS'],
                    bins=[-1, 0, 2, 5, 10, 50],
                    labels=[0, 1, 2, 3, 4]
                ).astype(float).fillna(0)

            # Ratios financiers
            if "AGE_YEARS" in df_engineered.columns and "EMPLOYMENT_YEARS" in df_engineered.columns:
                df_engineered['AGE_EMPLOYMENT_RATIO'] = df_engineered['AGE_YEARS'] / (df_engineered['EMPLOYMENT_YEARS'] + 1)

            if "AMT_CREDIT" in df_engineered.columns and "AMT_INCOME_TOTAL" in df_engineered.columns:
                df_engineered['CREDIT_INCOME_RATIO'] = df_engineered['AMT_CREDIT'] / df_engineered['AMT_INCOME_TOTAL']

            if "AMT_ANNUITY" in df_engineered.columns and "AMT_INCOME_TOTAL" in df_engineered.columns:
                df_engineered['ANNUITY_INCOME_RATIO'] = df_engineered['AMT_ANNUITY'] / df_engineered['AMT_INCOME_TOTAL']

            if "AMT_CREDIT" in df_engineered.columns and "AMT_GOODS_PRICE" in df_engineered.columns:
                df_engineered['CREDIT_GOODS_RATIO'] = df_engineered['AMT_CREDIT'] / (df_engineered['AMT_GOODS_PRICE'] + 1)

            if "AMT_ANNUITY" in df_engineered.columns and "AMT_CREDIT" in df_engineered.columns:
                df_engineered['ANNUITY_CREDIT_RATIO'] = df_engineered['AMT_ANNUITY'] / (df_engineered['AMT_CREDIT'] + 1)
                df_engineered['CREDIT_DURATION'] = df_engineered['AMT_CREDIT'] / (df_engineered['AMT_ANNUITY'] + 1)

            # Variables par personne
            if "AMT_INCOME_TOTAL" in df_engineered.columns and "CNT_FAM_MEMBERS" in df_engineered.columns:
                df_engineered['INCOME_PER_PERSON'] = df_engineered['AMT_INCOME_TOTAL'] / (df_engineered['CNT_FAM_MEMBERS'] + 1)

            if "AMT_CREDIT" in df_engineered.columns and "CNT_FAM_MEMBERS" in df_engineered.columns:
                df_engineered['CREDIT_PER_PERSON'] = df_engineered['AMT_CREDIT'] / (df_engineered['CNT_FAM_MEMBERS'] + 1)

            # Groupes de revenus et crédit
            if "AMT_INCOME_TOTAL" in df_engineered.columns:
                df_engineered['INCOME_GROUP'] = pd.cut(
                    df_engineered['AMT_INCOME_TOTAL'],
                    bins=[0, 100000, 200000, 500000, 1000000, np.inf],
                    labels=[0, 1, 2, 3, 4]
                ).astype(float).fillna(1)

            if "AMT_CREDIT" in df_engineered.columns:
                df_engineered['CREDIT_GROUP'] = pd.cut(
                    df_engineered['AMT_CREDIT'],
                    bins=[0, 200000, 500000, 1000000, 2000000, np.inf],
                    labels=[0, 1, 2, 3, 4]
                ).astype(float).fillna(1)

            # Indicateurs de propriété
            if "FLAG_OWN_CAR" in df_engineered.columns and "FLAG_OWN_REALTY" in df_engineered.columns:
                df_engineered['OWNS_PROPERTY'] = (df_engineered['FLAG_OWN_CAR'] * df_engineered['FLAG_OWN_REALTY']).astype(int)
                df_engineered['OWNS_NEITHER'] = ((1 - df_engineered['FLAG_OWN_CAR']) * (1 - df_engineered['FLAG_OWN_REALTY'])).astype(int)

            # Scores agrégés
            contact_features = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
                               'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
            available_contact = [f for f in contact_features if f in df_engineered.columns]
            if available_contact:
                df_engineered['CONTACT_SCORE'] = df_engineered[available_contact].sum(axis=1)

            # Score de documents
            doc_features = [f'FLAG_DOCUMENT_{i}' for i in range(2, 22)]
            available_docs = [f for f in doc_features if f in df_engineered.columns]
            if available_docs:
                df_engineered['DOCUMENT_SCORE'] = df_engineered[available_docs].sum(axis=1)

            # Score de région normalisé
            if "REGION_RATING_CLIENT" in df_engineered.columns:
                df_engineered['REGION_SCORE_NORMALIZED'] = 4 - df_engineered['REGION_RATING_CLIENT']

            # Features externes
            external_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
            available_ext = [f for f in external_features if f in df_engineered.columns]
            if available_ext:
                df_engineered['EXT_SOURCES_MEAN'] = df_engineered[available_ext].mean(axis=1)
                df_engineered['EXT_SOURCES_MAX'] = df_engineered[available_ext].max(axis=1)
                df_engineered['EXT_SOURCES_MIN'] = df_engineered[available_ext].min(axis=1)
                df_engineered['EXT_SOURCES_STD'] = df_engineered[available_ext].std(axis=1).fillna(0)
                df_engineered['EXT_SOURCES_COUNT'] = df_engineered[available_ext].count(axis=1)

                # Interaction âge et sources externes
                if "AGE_YEARS" in df_engineered.columns:
                    df_engineered['AGE_EXT_SOURCES_INTERACTION'] = df_engineered['AGE_YEARS'] * df_engineered['EXT_SOURCES_MEAN']

            # Features manquantes (indicateurs)
            missing_features = [
                ('AMT_ANNUITY_MISSING', 'AMT_ANNUITY'),
                ('AMT_GOODS_PRICE_MISSING', 'AMT_GOODS_PRICE'),
                ('DAYS_EMPLOYED_MISSING', 'DAYS_EMPLOYED'),
                ('CNT_FAM_MEMBERS_MISSING', 'CNT_FAM_MEMBERS'),
                ('DAYS_REGISTRATION_MISSING', 'DAYS_REGISTRATION')
            ]

            for missing_col, original_col in missing_features:
                if original_col in df_engineered.columns:
                    df_engineered[missing_col] = df_engineered[original_col].isnull().astype(int)
                else:
                    df_engineered[missing_col] = 0

            # 5. Nettoyage final des données
            # Remplacer les valeurs infinies
            df_engineered = df_engineered.replace([np.inf, -np.inf], 0)

            # Convertir toutes les colonnes en numérique
            for col in df_engineered.columns:
                if df_engineered[col].dtype == "object":
                    try:
                        df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce')
                    except (ValueError, TypeError):
                        df_engineered[col] = 0

            # Remplir les valeurs manquantes
            df_engineered = df_engineered.fillna(0)

            # 6. Ordre final des features (les 153 attendues)
            final_feature_order = [
                'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE',
                'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
                'OWN_CAR_AGE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
                'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'OCCUPATION_TYPE',
                'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
                'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
                'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
                'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE',
                'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG',
                'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG',
                'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG',
                'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG',
                'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE',
                'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',
                'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE',
                'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
                'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI',
                'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
                'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI',
                'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI',
                'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'FONDKAPREMONT_MODE',
                'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',
                'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
                'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2',
                'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
                'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',
                'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
                'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
                'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR',
                'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'AGE_YEARS',
                'EMPLOYMENT_YEARS', 'DAYS_EMPLOYED_ABNORMAL', 'YEARS_SINCE_REGISTRATION',
                'YEARS_SINCE_ID_PUBLISH', 'AGE_GROUP', 'EMPLOYMENT_GROUP', 'AGE_EMPLOYMENT_RATIO',
                'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'CREDIT_GOODS_RATIO',
                'ANNUITY_CREDIT_RATIO', 'CREDIT_DURATION', 'INCOME_PER_PERSON', 'CREDIT_PER_PERSON',
                'INCOME_GROUP', 'CREDIT_GROUP', 'OWNS_PROPERTY', 'OWNS_NEITHER', 'CONTACT_SCORE',
                'DOCUMENT_SCORE', 'REGION_SCORE_NORMALIZED', 'EXT_SOURCES_MEAN', 'EXT_SOURCES_MAX',
                'EXT_SOURCES_MIN', 'EXT_SOURCES_STD', 'EXT_SOURCES_COUNT', 'AGE_EXT_SOURCES_INTERACTION',
                'AMT_ANNUITY_MISSING', 'AMT_GOODS_PRICE_MISSING', 'DAYS_EMPLOYED_MISSING',
                'CNT_FAM_MEMBERS_MISSING', 'DAYS_REGISTRATION_MISSING'
            ]

            # S'assurer que toutes les features finales sont présentes
            for feature in final_feature_order:
                if feature not in df_engineered.columns:
                    df_engineered[feature] = 0

            # Réorganiser les colonnes dans l'ordre attendu
            df_final = df_engineered[final_feature_order]

            logger.info(f"Feature engineering terminé. {len(df_final.columns)} features finales")
            return df_final
                "FLAG_OWN_CAR": {"Y": 1, "N": 0, "Oui": 1, "Non": 0},
                "FLAG_OWN_REALTY": {"Y": 1, "N": 0, "Oui": 1, "Non": 0},
                "NAME_EDUCATION_TYPE": {
                    "Secondary / secondary special": 0,
                    "Higher education": 1,
                    "Incomplete higher": 2,
                    "Lower secondary": 3,
                    "Academic degree": 4
                },
                "NAME_FAMILY_STATUS": {
                    "Single / not married": 0,
                    "Married": 1,
                    "Civil marriage": 2,
                    "Separated": 3,
                    "Widow": 4
                },
                "NAME_HOUSING_TYPE": {
                    "House / apartment": 0,
                    "With parents": 1,
                    "Municipal apartment": 2,
                    "Rented apartment": 3,
                    "Office / cooperative apartment": 4,
                    "Co-op apartment": 5
                },
                "NAME_INCOME_TYPE": {
                    "Working": 0,
                    "Commercial associate": 1,
                    "Pensioner": 2,
                    "State servant": 3,
                    "Student": 4,
                    "Unemployed": 5,
                    "Businessman": 6,
                    "Maternity leave": 7
                },
                "NAME_TYPE_SUITE": {
                    "Unaccompanied": 0,
                    "Family": 1,
                    "Spouse, partner": 2,
                    "Children": 3,
                    "Other_B": 4,
                    "Other_A": 5,
                    "Group of people": 6
                }
            }

            # Appliquer les mappings
            for col, mapping in categorical_mappings.items():
                if col in df_engineered.columns:
                    # Utiliser replace au lieu de map pour éviter les erreurs de type
                    df_engineered[col] = df_engineered[col].replace(mapping).fillna(0)

            # 3. Traitement des variables temporelles
            if "DAYS_BIRTH" in df_engineered.columns:
                df_engineered["DAYS_BIRTH"] = pd.Series(pd.to_numeric(df_engineered["DAYS_BIRTH"], errors='coerce')).fillna(0).astype(int)

            if "DAYS_EMPLOYED" in df_engineered.columns:
                df_engineered["DAYS_EMPLOYED"] = pd.Series(pd.to_numeric(df_engineered["DAYS_EMPLOYED"], errors='coerce')).fillna(0).astype(int)
                # Traitement des valeurs aberrantes
                df_engineered["DAYS_EMPLOYED"] = df_engineered["DAYS_EMPLOYED"].replace(365243, 0)

            if "DAYS_REGISTRATION" in df_engineered.columns:
                df_engineered["DAYS_REGISTRATION"] = pd.Series(pd.to_numeric(df_engineered["DAYS_REGISTRATION"], errors='coerce')).fillna(0).astype(int)

            if "DAYS_ID_PUBLISH" in df_engineered.columns:
                df_engineered["DAYS_ID_PUBLISH"] = pd.Series(pd.to_numeric(df_engineered["DAYS_ID_PUBLISH"], errors='coerce')).fillna(0).astype(int)

            # 4. Traitement des variables numériques
            numeric_columns = ["CNT_CHILDREN", "CNT_FAM_MEMBERS", "AMT_INCOME_TOTAL",
                             "AMT_ANNUITY", "AMT_CREDIT", "AMT_GOODS_PRICE"]

            for col in numeric_columns:
                if col in df_engineered.columns:
                    if col.startswith("CNT_"):
                        df_engineered[col] = pd.Series(pd.to_numeric(df_engineered[col], errors='coerce')).fillna(0).astype(int)
                    else:
                        df_engineered[col] = pd.Series(pd.to_numeric(df_engineered[col], errors='coerce')).fillna(0).astype(float)

            # 5. S'assurer que toutes les features d'entraînement sont présentes
            for feature in self.training_features:
                if feature not in df_engineered.columns:
                    # Valeur par défaut selon le type de feature
                    if "FLAG_" in feature:
                        df_engineered[feature] = 0
                    elif "AMT_" in feature or "AVG" in feature or "MEDI" in feature or "MODE" in feature:
                        df_engineered[feature] = 0.0
                    elif "CNT_" in feature or "DAYS_" in feature or "HOUR_" in feature:
                        df_engineered[feature] = 0
                    elif "RATE_" in feature or "RATING_" in feature:
                        df_engineered[feature] = 2
                    elif "TYPE" in feature or "STATUS" in feature or "SUITE" in feature:
                        df_engineered[feature] = 0
                    else:
                        df_engineered[feature] = 0

                    logger.info(f"Feature manquante ajoutée avec valeur par défaut: {feature}")

            # 6. Validation et cohérence des données
            # Vérifier la cohérence des montants
            if "AMT_CREDIT" in df_engineered.columns and "AMT_ANNUITY" in df_engineered.columns:
                # S'assurer que l'annuité n'est pas supérieure au crédit
                mask = (df_engineered["AMT_ANNUITY"] > df_engineered["AMT_CREDIT"]) & (df_engineered["AMT_CREDIT"] > 0)
                if mask.any():
                    logger.warning("Annuité supérieure au montant du crédit détectée - correction appliquée")
                    df_engineered.loc[mask, "AMT_ANNUITY"] = df_engineered.loc[mask, "AMT_CREDIT"] * 0.1

            # Vérifier la cohérence des âges
            if "DAYS_BIRTH" in df_engineered.columns and "DAYS_EMPLOYED" in df_engineered.columns:
                # S'assurer que l'expérience professionnelle ne dépasse pas l'âge
                age_years = -df_engineered["DAYS_BIRTH"] / 365.25
                employment_years = -df_engineered["DAYS_EMPLOYED"] / 365.25
                mask = (employment_years > age_years - 18) & (age_years > 18)
                if mask.any():
                    logger.warning("Expérience professionnelle incohérente avec l'âge - correction appliquée")
                    df_engineered.loc[mask, "DAYS_EMPLOYED"] = -(age_years[mask] - 18) * 365.25

            # 7. Nettoyage final des données
            # Remplacer les valeurs infinies par des valeurs finies
            df_engineered = df_engineered.replace([np.inf, -np.inf], np.nan)

            # S'assurer que toutes les colonnes sont numériques AVANT le fillna
            for col in df_engineered.columns:
                if df_engineered[col].dtype == "object":
                    try:
                        df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce')
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Impossible de convertir la colonne {col} en numérique: {e}")
                        df_engineered[col] = 0

            # Remplir les valeurs manquantes restantes APRÈS la conversion
            df_engineered = df_engineered.fillna(0)

            # 6. Garder seulement les features d'entraînement
            final_features = [col for col in self.training_features if col in df_engineered.columns]
            if len(final_features) == 1:
                # Si une seule feature, créer un DataFrame avec une colonne
                df_final = pd.DataFrame({final_features[0]: df_engineered[final_features[0]]})
            else:
                df_final = df_engineered[final_features]

            logger.info(f"Feature engineering terminé. {len(final_features)} features finales")
            # Type casting explicite pour mypy
            if isinstance(df_final, pd.Series):
                return pd.DataFrame(df_final).T
            else:
                return df_final

        except Exception as e:
            logger.error(f"Erreur lors du feature engineering: {e}")
            raise

    def get_training_features(self) -> List[str]:
        """Retourne la liste des features d'entraînement"""
        return self.training_features.copy()

    def get_features_to_remove(self) -> List[str]:
        """Retourne la liste des features à supprimer"""
        return self.features_to_remove.copy()

# Fonction de convenance
def create_features_complete(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction de convenance pour créer toutes les features

    Args:
        df: DataFrame avec les données brutes

    Returns:
        DataFrame avec les features engineering appliquées
    """
    engineer = CreditFeatureEngineer()
    return engineer.engineer_features(df)
