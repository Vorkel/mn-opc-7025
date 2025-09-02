"""
Module de feature engineering COMPLET pour le scoring crédit
Préserve TOUTES les features originales + ajoute les features dérivées
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CreditFeatureEngineerComplete:
    """
    Classe pour l'ingénierie des features de scoring crédit COMPLÈTE
    """
    
    def __init__(self) -> None:
        """Initialise l'ingénieur de features"""
        self.categorical_mappings = {
            "CODE_GENDER": {"M": 1, "F": 0},
            "FLAG_OWN_CAR": {"Y": 1, "N": 0},
            "FLAG_OWN_REALTY": {"Y": 1, "N": 0},
            "NAME_INCOME_TYPE": {
                "Working": 0, "Salarié CDI": 0, "Salarié CDD": 0,
                "Commercial associate": 1, "Associé commercial": 1,
                "Pensioner": 2, "Retraité": 2,
                "State servant": 3, "Fonctionnaire": 3,
                "Unemployed": 4, "Chômeur": 4,
                "Student": 5, "Étudiant": 5,
                "Businessman": 6, "Indépendant": 6,
                "Maternity leave": 7,
            },
            "NAME_EDUCATION_TYPE": {
                "Secondary / secondary special": 0, "Secondaire": 0,
                "Higher education": 1, "Supérieur": 1,
                "Incomplete higher": 2, "Supérieur incomplet": 2,
                "Lower secondary": 3, "Secondaire inférieur": 3,
                "Academic degree": 4, "Diplôme universitaire": 4,
            },
            "NAME_FAMILY_STATUS": {
                "Single / not married": 0, "Célibataire": 0,
                "Married": 1, "Marié": 1,
                "Civil marriage": 2, "Union civile": 2,
                "Widow": 3, "Veuf/Veuve": 3,
                "Separated": 4, "Séparé": 4,
                "Unknown": 5,
            },
            "NAME_HOUSING_TYPE": {
                "House / apartment": 0, "Propriétaire": 0, "Maison/Appartement": 0,
                "Rented apartment": 1, "Locataire": 1,
                "With parents": 2, "Chez les parents": 2,
                "Municipal apartment": 3, "Appartement municipal": 3,
                "Office apartment": 4, "Appartement de fonction": 4,
                "Co-op apartment": 5, "Appartement coopératif": 5,
            },
        }
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applique l'ingénierie des features complète en préservant TOUTES les features originales
        
        Args:
            df: DataFrame avec les données brutes
            
        Returns:
            DataFrame avec toutes les features calculées
        """
        try:
            # IMPORTANT: Préserver TOUTES les features originales
            df_engineered = df.copy()
            
            # Liste des features originales importantes à préserver
            original_features_to_preserve = [
                # Features APARTMENTS
                "APARTMENTS_AVG", "APARTMENTS_MEDI", "APARTMENTS_MODE",
                "LIVINGAPARTMENTS_AVG", "LIVINGAPARTMENTS_MEDI", "LIVINGAPARTMENTS_MODE",
                "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAPARTMENTS_MODE",
                
                # Features BASEMENTAREA
                "BASEMENTAREA_AVG", "BASEMENTAREA_MEDI", "BASEMENTAREA_MODE",
                
                # Features FLAG_DOCUMENT (toutes)
                "FLAG_DOCUMENT_1", "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4",
                "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8",
                "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12",
                "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
                "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20",
                "FLAG_DOCUMENT_21",
                
                # Features AMT_REQ_CREDIT_BUREAU
                "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
                "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
                "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"
            ]
            
            # S'assurer que toutes les features importantes sont présentes
            for feature in original_features_to_preserve:
                if feature not in df_engineered.columns:
                    # Valeur par défaut selon le type de feature
                    if "FLAG_" in feature:
                        df_engineered[feature] = 0  # Flag = 0 par défaut
                    elif "AMT_" in feature or "AVG" in feature or "MEDI" in feature or "MODE" in feature:
                        df_engineered[feature] = 0.0  # Valeur numérique = 0.0 par défaut
                    else:
                        df_engineered[feature] = 0  # Autre = 0 par défaut
            
            # =============================================================================
            # FEATURE ENGINEERING - Variables temporelles
            # =============================================================================
            
            # Convertir les jours en années
            df_engineered["AGE_YEARS"] = -df_engineered["DAYS_BIRTH"] / 365.25
            df_engineered["EMPLOYMENT_YEARS"] = -df_engineered["DAYS_EMPLOYED"] / 365.25
            
            # Nettoyer les valeurs aberrantes DAYS_EMPLOYED
            df_engineered["DAYS_EMPLOYED_ABNORMAL"] = (
                df_engineered["DAYS_EMPLOYED"] == 365243
            ).astype(int)
            df_engineered["DAYS_EMPLOYED"] = df_engineered["DAYS_EMPLOYED"].replace(365243, np.nan)
            df_engineered["EMPLOYMENT_YEARS"] = -df_engineered["DAYS_EMPLOYED"] / 365.25
            
            # Variables temporelles supplémentaires
            df_engineered["YEARS_SINCE_REGISTRATION"] = -df_engineered["DAYS_REGISTRATION"] / 365.25
            df_engineered["YEARS_SINCE_ID_PUBLISH"] = -df_engineered["DAYS_ID_PUBLISH"] / 365.25
            
            # Groupes d'âge
            age_groups = pd.cut(
                df_engineered["AGE_YEARS"],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=[0, 1, 2, 3, 4, 5],
            )
            df_engineered["AGE_GROUP"] = pd.Categorical(age_groups).codes
            
            # Groupes d'expérience
            emp_groups = pd.cut(
                df_engineered["EMPLOYMENT_YEARS"],
                bins=[-1, 0, 2, 5, 10, 20, 50],
                labels=[0, 1, 2, 3, 4, 5],
            )
            df_engineered["EMPLOYMENT_GROUP"] = pd.Categorical(emp_groups).codes
            
            # Ratios temporels
            df_engineered["AGE_EMPLOYMENT_RATIO"] = (
                df_engineered["AGE_YEARS"] / (df_engineered["EMPLOYMENT_YEARS"] + 1)
            )
            
            # =============================================================================
            # FEATURE ENGINEERING - Variables financières
            # =============================================================================
            
            # Ratios principaux
            df_engineered["CREDIT_INCOME_RATIO"] = (
                df_engineered["AMT_CREDIT"] / df_engineered["AMT_INCOME_TOTAL"]
            )
            df_engineered["ANNUITY_INCOME_RATIO"] = (
                df_engineered["AMT_ANNUITY"] / df_engineered["AMT_INCOME_TOTAL"]
            )
            df_engineered["CREDIT_GOODS_RATIO"] = (
                df_engineered["AMT_CREDIT"] / df_engineered["AMT_GOODS_PRICE"]
            )
            df_engineered["ANNUITY_CREDIT_RATIO"] = (
                df_engineered["AMT_ANNUITY"] / df_engineered["AMT_CREDIT"]
            )
            
            # Durée estimée du crédit
            df_engineered["CREDIT_DURATION"] = (
                df_engineered["AMT_CREDIT"] / df_engineered["AMT_ANNUITY"]
            )
            
            # Revenus et crédits par personne
            df_engineered["INCOME_PER_PERSON"] = (
                df_engineered["AMT_INCOME_TOTAL"] / df_engineered["CNT_FAM_MEMBERS"]
            )
            df_engineered["CREDIT_PER_PERSON"] = (
                df_engineered["AMT_CREDIT"] / df_engineered["CNT_FAM_MEMBERS"]
            )
            
            # Groupes de revenus
            income_groups = pd.cut(
                df_engineered["AMT_INCOME_TOTAL"],
                bins=[0, 100000, 200000, 300000, 500000, np.inf],
                labels=[0, 1, 2, 3, 4],
            )
            df_engineered["INCOME_GROUP"] = pd.Categorical(income_groups).codes
            
            # Groupes de crédit
            credit_groups = pd.cut(
                df_engineered["AMT_CREDIT"],
                bins=[0, 200000, 500000, 1000000, 2000000, np.inf],
                labels=[0, 1, 2, 3, 4],
            )
            df_engineered["CREDIT_GROUP"] = pd.Categorical(credit_groups).codes
            
            # Indicateurs de richesse
            df_engineered["OWNS_PROPERTY"] = (
                (df_engineered["FLAG_OWN_CAR"] == "Y") & 
                (df_engineered["FLAG_OWN_REALTY"] == "Y")
            ).astype(int)
            df_engineered["OWNS_NEITHER"] = (
                (df_engineered["FLAG_OWN_CAR"] == "N") & 
                (df_engineered["FLAG_OWN_REALTY"] == "N")
            ).astype(int)
            
            # =============================================================================
            # FEATURE ENGINEERING - Variables d'agrégation
            # =============================================================================
            
            # Scores de contact
            contact_features = [
                "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", 
                "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"
            ]
            df_engineered["CONTACT_SCORE"] = df_engineered[contact_features].sum(axis=1)
            
            # Scores de documents
            doc_features = [
                col for col in df_engineered.columns if col.startswith("FLAG_DOCUMENT_")
            ]
            if doc_features:
                df_engineered["DOCUMENT_SCORE"] = df_engineered[doc_features].sum(axis=1)
            else:
                df_engineered["DOCUMENT_SCORE"] = 0
                
            # Score de région normalisé
            df_engineered["REGION_SCORE_NORMALIZED"] = 4 - df_engineered["REGION_RATING_CLIENT"]
            
            # Features externes (EXT_SOURCE) - valeurs par défaut si non présentes
            df_engineered["EXT_SOURCE_1"] = df_engineered.get("EXT_SOURCE_1", 0.5)
            df_engineered["EXT_SOURCE_2"] = df_engineered.get("EXT_SOURCE_2", 0.5)
            df_engineered["EXT_SOURCE_3"] = df_engineered.get("EXT_SOURCE_3", 0.5)
            
            # Calculer les agrégations EXT_SOURCE
            ext_sources = [
                df_engineered["EXT_SOURCE_1"], 
                df_engineered["EXT_SOURCE_2"], 
                df_engineered["EXT_SOURCE_3"]
            ]
            df_engineered["EXT_SOURCES_MEAN"] = np.mean(ext_sources, axis=0)
            df_engineered["EXT_SOURCES_MAX"] = np.max(ext_sources, axis=0)
            df_engineered["EXT_SOURCES_MIN"] = np.min(ext_sources, axis=0)
            df_engineered["EXT_SOURCES_STD"] = np.std(ext_sources, axis=0)
            df_engineered["EXT_SOURCES_COUNT"] = 3
            
            # Interactions
            df_engineered["AGE_EXT_SOURCES_INTERACTION"] = (
                df_engineered["AGE_YEARS"] * df_engineered["EXT_SOURCES_MEAN"]
            )
            
            # =============================================================================
            # Gestion des valeurs manquantes
            # =============================================================================
            
            # Créer des indicateurs pour les features importantes
            important_features = [
                "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_EMPLOYED", 
                "CNT_FAM_MEMBERS", "DAYS_REGISTRATION"
            ]
            for feature in important_features:
                if feature in df_engineered.columns:
                    indicator_name = f"{feature}_MISSING"
                    df_engineered[indicator_name] = df_engineered[feature].isnull().astype(int)
            
            # Imputation par type
            numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_engineered[col].isnull().sum() > 0:
                    median_val = df_engineered[col].median()
                    df_engineered[col] = df_engineered[col].fillna(median_val)
            
            # =============================================================================
            # Encodage des variables catégorielles
            # =============================================================================
            
            # Appliquer les mappings
            for col, mapping in self.categorical_mappings.items():
                if col in df_engineered.columns:
                    df_engineered[col] = (
                        df_engineered[col].astype(str).map(lambda x: mapping.get(x, 0)).fillna(0)
                    )

            # Traitement des variables catégorielles restantes (encodage simple)
            for col in df_engineered.select_dtypes(include=["object"]).columns:
                df_engineered[col] = pd.Categorical(df_engineered[col]).codes

            # Gestion des valeurs manquantes finales
            df_engineered = df_engineered.fillna(0)
            
            # S'assurer que toutes les colonnes sont numériques
            for col in df_engineered.columns:
                if df_engineered[col].dtype == "object":
                    # Conversion explicite en Series pandas pour éviter les problèmes de typage
                    df_engineered[col] = pd.Series(
                        pd.to_numeric(df_engineered[col], errors="coerce")
                    ).fillna(0)
            
            logger.info(f"Feature engineering terminé: {len(df_engineered.columns)} features créées")
            return df_engineered
            
        except Exception as e:
            logger.error(f"Erreur lors du feature engineering: {e}")
            raise


def create_features_complete(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction utilitaire pour créer les features complètes
    
    Args:
        df: DataFrame avec les données brutes
        
    Returns:
        DataFrame avec toutes les features calculées
    """
    engineer = CreditFeatureEngineerComplete()
    return engineer.engineer_features(df)
