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
        Applique l'ingénierie des features standard Home Credit

        Args:
            df: DataFrame avec les données brutes

        Returns:
            DataFrame avec les features engineering appliquées
        """
        try:
            logger.info("Début de l'ingénierie des features")
            df_engineered = df.copy()

            # 1. Supprimer les features fictives si présentes
            for feature in self.features_to_remove:
                if feature in df_engineered.columns:
                    df_engineered = df_engineered.drop(columns=[feature])
                    logger.info(f"Feature fictive supprimée: {feature}")

            # 2. Encodage des variables catégorielles standard
            categorical_mappings = {
                "CODE_GENDER": {"M": 1, "F": 0, "Homme": 1, "Femme": 0},
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
