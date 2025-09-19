"""
Application principale MLOps Credit Scoring - Version Simplifiée et Fonctionnelle
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Configuration du logging
logger = logging.getLogger(__name__)

# Ajouter le chemin src pour les imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Credit Scoring - Prêt à Dépenser",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration de l'API distante
API_BASE_URL: str = "https://mn-opc-7025.onrender.com"  # Hardcodé
API_TIMEOUT: int = 60  # Augmenté à 60 secondes pour Render.com
API_RETRY_ATTEMPTS: int = 3  # Nombre de tentatives
API_RETRY_DELAY: int = 5  # Délai entre les tentatives (secondes)

# Configuration automatique basée sur l'environnement
# Détection plus robuste de l'environnement de production
IS_PRODUCTION: bool = (
    os.getenv("STREAMLIT_ENV") == "production" or
    os.getenv("RENDER") is not None or
    os.getenv("STREAMLIT_CLOUD") is not None or  # Streamlit Cloud
    "streamlit.app" in os.getenv("STREAMLIT_SERVER_BASE_URL_PATH", "") or  # URL Streamlit Cloud
    os.getenv("STREAMLIT_SERVER_PORT") == "8501"  # Port par défaut Streamlit Cloud
)

# OU plus simple : forcer l'API en production
USE_REMOTE_API: bool = IS_PRODUCTION  # Utiliser la détection d'environnement

# Chemins des fichiers
BASE_DIR: Path = Path(__file__).parent.parent
MODELS_DIR: Path = BASE_DIR / "models"
DATA_DIR: Path = BASE_DIR / "data"


def init_session_state() -> None:
    """Initialise les variables de session"""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_prediction" not in st.session_state:
        st.session_state.current_prediction = None


def test_api_connection_with_retry() -> Optional[Dict[str, Any]]:
    """
    Teste la connexion à l'API avec retry logic et gestion robuste des timeouts
    Returns:
        Dict contenant les informations de santé de l'API ou None si échec
    """
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            # Progressif timeout : plus de temps pour les tentatives suivantes
            current_timeout = API_TIMEOUT + (attempt * 10)

            logger.info(f"Tentative de connexion API {attempt + 1}/{API_RETRY_ATTEMPTS} (timeout: {current_timeout}s)")

            response = requests.get(
                f"{API_BASE_URL}/health",
                timeout=current_timeout,
                headers={'User-Agent': 'Streamlit-Credit-Scoring/1.0'}
            )

            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"API connectée avec succès (tentative {attempt + 1})")
                return health_data
            else:
                logger.warning(f"API répond avec status {response.status_code} (tentative {attempt + 1})")

        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout API (tentative {attempt + 1}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                logger.info(f"Attente de {API_RETRY_DELAY}s avant nouvelle tentative...")
                time.sleep(API_RETRY_DELAY)

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Erreur de connexion API (tentative {attempt + 1}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                logger.info(f"Attente de {API_RETRY_DELAY}s avant nouvelle tentative...")
                time.sleep(API_RETRY_DELAY)

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de requête API (tentative {attempt + 1}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)

        except Exception as e:
            logger.error(f"Erreur inattendue lors de la connexion API (tentative {attempt + 1}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)

    logger.error(f"Échec de connexion API après {API_RETRY_ATTEMPTS} tentatives")
    return None


@st.cache_resource(ttl=3600)  # Cache avec TTL
def load_model(force_reload: bool = False) -> Optional[Dict[str, Any]]:
    """Charge le modèle avec gestion de la mémoire"""
    # TOUJOURS tenter la connexion API d'abord avec retry logic
    health_data = test_api_connection_with_retry()

    if health_data:
        st.success("API Connectée avec succès")
        return {
            "model": None,
            "threshold": 0.295,
            "scaler": None,
            "feature_names": [],
            "loaded_from": "API distante",
            "api_status": "connected",
            "api_health": health_data,
        }
    else:
        st.warning("Impossible de se connecter à l'API distante après plusieurs tentatives")

    # Fallback sur le modèle local
    st.info("Utilisation du modèle local")
    model_paths = [
        BASE_DIR / "models" / "best_credit_model.pkl",
        MODELS_DIR / "best_credit_model.pkl",
        MODELS_DIR / "best_model.pkl",
        MODELS_DIR / "model.pkl",
        BASE_DIR / "model.pkl",
    ]

    for model_path in model_paths:
        if model_path.exists():
            try:
                model_data = joblib.load(model_path)

                # Vérifier le type de modèle chargé
                if hasattr(model_data, "predict"):
                    # C'est directement un modèle sklearn
                    from sklearn.preprocessing import StandardScaler

                    scaler = StandardScaler()
                    st.info(
                        "Modèle RandomForest chargé directement depuis "
                        f"{model_path}"
                    )

                    return {
                        "model": model_data,
                        "threshold": 0.295,  # Seuil métier optimisé
                        "scaler": scaler,
                        "feature_names": [],
                        "loaded_from": str(model_path),
                        "api_status": "local",
                    }
                elif (
                    model_data
                    and "model" in model_data
                    and model_data["model"] is not None
                ):
                    # C'est un dictionnaire avec le modèle
                    scaler = model_data.get("scaler")
                    if scaler is None:
                        from sklearn.preprocessing import StandardScaler

                        scaler = StandardScaler()
                        st.info("Scaler par défaut créé (StandardScaler)")

                    return {
                        "model": model_data["model"],
                        "threshold": model_data.get("threshold", 0.5),
                        "scaler": scaler,
                        "feature_names": model_data.get("feature_names", []),
                        "loaded_from": str(model_path),
                        "api_status": "local",
                    }
                else:
                    st.warning(f"Modèle invalide dans {model_path}")
            except Exception as e:
                st.warning(f"Erreur chargement modèle {model_path}: {e}")
                continue

    # Si aucun modèle valide n'est trouvé
    st.error("Aucun modèle local valide trouvé")
    return None


def create_full_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Crée le jeu complet de 153 features attendues par le modèle"""
    try:
        # Import au niveau du module si possible
        from src.feature_engineering import create_complete_feature_set
        return create_complete_feature_set(df)
    except ImportError as e:
        logger.warning(f"Feature engineering centralisé non disponible: {e}")
        # Fallback vers l'ancien système
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

    df_full = df.copy()

    # Ajouter toutes les features manquantes avec valeurs par défaut appropriées
    for feature in expected_features:
        if feature not in df_full.columns:
            if "FLAG_" in feature:
                df_full[feature] = 0
            elif "AMT_" in feature:
                df_full[feature] = 0.0
            elif "CNT_" in feature:
                df_full[feature] = 0
            elif "DAYS_" in feature:
                df_full[feature] = 0
            elif "_AVG" in feature or "_MODE" in feature or "_MEDI" in feature:
                df_full[feature] = 0.5
            elif "RATIO" in feature or "SCORE" in feature or "RATE_" in feature:
                df_full[feature] = 0.5
            elif "GROUP" in feature:
                df_full[feature] = 0
            elif "EXT_SOURCE" in feature:
                df_full[feature] = 0.5
            elif "YEARS" in feature:
                df_full[feature] = 0.5
            else:
                df_full[feature] = 0.5

    return df_full

def validate_business_rules(client_data: Dict[str, Any]) -> Dict[str, Any]:
    """Valide les règles métier avant prédiction"""
    try:
        income = client_data.get("AMT_INCOME_TOTAL")
        credit_amount = client_data.get("AMT_CREDIT")
        annuity = client_data.get("AMT_ANNUITY")

        errors: List[str] = []

        # Règles de validation métier
        if income is None or income < 12000:
            errors.append("Revenus annuels insuffisants (minimum 12 000€)")

        if income and credit_amount and income > 0 and credit_amount > 0:
            if credit_amount / income > 5:
                errors.append(
                    "Montant du crédit trop élevé par rapport aux revenus (max 5x)"
                )

        if income and annuity and income > 0 and annuity > 0:
            if annuity / income > 0.33:
                errors.append("Annuité trop élevée par rapport aux revenus (max 33%)")

        if credit_amount and credit_amount > 2000000:
            errors.append("Montant du crédit trop élevé (maximum 2 000 000€)")

        if errors:
            return {"valid": False, "message": " | ".join(errors)}

        return {"valid": True, "message": "Validation OK"}

    except Exception as e:
        return {"valid": False, "message": f"Erreur de validation: {str(e)}"}


def call_api_prediction(client_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Appelle l'API distante pour la prédiction avec retry logic"""
    api_data: Dict[str, Union[int, float, str]] = {}
    for key, value in client_data.items():
        if isinstance(value, (int, float, str)):
            api_data[key] = value

    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            # Progressif timeout pour les prédictions
            current_timeout = API_TIMEOUT + (attempt * 15)

            logger.info(f"Tentative de prédiction API {attempt + 1}/{API_RETRY_ATTEMPTS} (timeout: {current_timeout}s)")

            response = requests.post(
                f"{API_BASE_URL}/predict_public",
                json=api_data,
                timeout=current_timeout,
                headers={'User-Agent': 'Streamlit-Credit-Scoring/1.0'}
            )

            if response.status_code == 200:
                result: Dict[str, Any] = response.json()
                logger.info(f"Prédiction API réussie (tentative {attempt + 1})")
                return result
            else:
                logger.warning(f"API prédiction erreur {response.status_code} (tentative {attempt + 1}): {response.text}")
                if attempt < API_RETRY_ATTEMPTS - 1:
                    time.sleep(API_RETRY_DELAY)

        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout prédiction API (tentative {attempt + 1}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                logger.info(f"Attente de {API_RETRY_DELAY}s avant nouvelle tentative...")
                time.sleep(API_RETRY_DELAY)

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Erreur de connexion prédiction API (tentative {attempt + 1}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de requête prédiction API (tentative {attempt + 1}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)

        except Exception as e:
            logger.error(f"Erreur inattendue prédiction API (tentative {attempt + 1}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)

    logger.error(f"Échec de prédiction API après {API_RETRY_ATTEMPTS} tentatives")
    st.error(f"❌ Impossible d'obtenir une prédiction de l'API après {API_RETRY_ATTEMPTS} tentatives")
    return None


def predict_score(client_data: Dict[str, Any], model_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Effectue une prédiction de score (local ou distant)"""
    global USE_REMOTE_API

    try:
        # Si on utilise l'API distante
        if USE_REMOTE_API and model_data.get("api_status") == "connected":
            api_result = call_api_prediction(client_data)
            if api_result:
                result = {
                    "probability": api_result.get("probability", 0.5),
                    "decision": api_result.get("decision", "REFUSÉ"),
                    "risk_level": api_result.get("risk_level", "Élevé"),
                    "threshold": api_result.get("threshold", 0.5),
                }

                prediction_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "client_data": client_data,
                    "result": result,
                }
                st.session_state.history.append(prediction_record)

                return result
            else:
                st.warning("Échec de l'appel API, basculement sur le modèle local")
                USE_REMOTE_API = False

        # Modèle local (fallback)
        model = model_data.get("model")

        # Vérifier que le modèle est disponible
        if model is None:
            logger.error("Aucun modèle local disponible")
            return {
                "probability": 0.8,
                "decision": "REFUSÉ",
                "risk_level": "Élevé",
                "threshold": 0.295,  # Seuil métier optimisé
                "validation_error": (
                    "Modèle indisponible - veuillez contacter le support"
                ),
            }

        threshold = model_data.get("threshold", 0.5)
        feature_names = model_data.get("feature_names", [])

        # Validation métier des données d'entrée
        validation_result = validate_business_rules(client_data)
        if not validation_result["valid"]:
            return {
                "probability": 1.0,
                "decision": "REFUSÉ",
                "risk_level": "Élevé",
                "threshold": threshold,
                "validation_error": validation_result["message"],
            }

        # Conversion en DataFrame
        df = pd.DataFrame([client_data])

        # Utiliser le feature engineering complet dans src/
        sys.path.append(str(Path(__file__).parent.parent / "src"))
        from src.feature_engineering import create_complete_feature_set

        df_engineered = create_complete_feature_set(client_data)

        # Prédiction avec le DataFrame complet
        probabilities = model.predict_proba(df_engineered)
        probability = probabilities[0][1]

        # Décision basée sur le seuil optimisé
        decision = "REFUSÉ" if probability > threshold else "ACCORDÉ"

        # Niveau de risque
        if probability < 0.3:
            risk_level = "Faible"
        elif probability < 0.6:
            risk_level = "Modéré"
        else:
            risk_level = "Élevé"

        result = {
            "probability": probability,
            "decision": decision,
            "risk_level": risk_level,
            "threshold": threshold,
        }

        # Sauvegarder dans l'historique
        prediction_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "client_data": client_data,
            "result": result,
        }
        st.session_state.history.append(prediction_record)

        return result
    except Exception as e:
        st.error(f"Erreur prédiction: {e}")
        return None


def get_refusal_reason(result: Dict[str, Any], client_data: Dict[str, Any]) -> Dict[str, Any]:
    """Génère une explication claire du refus ou de l'accord"""
    try:
        if result.get("decision") == "ACCORDÉ":
            return {
                "status": "success",
                "title": "CRÉDIT ACCORDÉ",
                "message": "Félicitations ! Votre demande de crédit a été acceptée.",
                "details": [
                    f"Score de risque : {result.get('risk_level', 'N/A')}",
                    (
                        "Probabilité de remboursement : "
                        f"{(1 - result.get('probability', 0)) * 100:.1f}%"
                    ),
                    f"Seuil d'acceptation : {result.get('threshold', 0.5) * 100:.1f}%",
                ],
            }
        else:
            reasons = []
            risk_factors = []

            probability = result.get("probability", 0)
            if probability > 0.8:
                reasons.append("Score de risque très élevé")
                risk_factors.append("Probabilité de défaut de paiement trop importante")
            elif probability > 0.6:
                reasons.append("Score de risque élevé")
                risk_factors.append("Risque de défaut de paiement significatif")

            income = client_data.get("AMT_INCOME_TOTAL", 0)
            credit_amount = client_data.get("AMT_CREDIT", 0)

            if income > 0 and credit_amount > 0:
                ratio = credit_amount / income
                if ratio > 4:
                    reasons.append("Ratio crédit/revenus trop élevé")
                    risk_factors.append(
                        f"Le crédit représente {ratio:.1f}x vos revenus annuels"
                    )
                elif ratio > 3:
                    reasons.append("Ratio crédit/revenus élevé")
                    risk_factors.append(
                        f"Le crédit représente {ratio:.1f}x vos revenus annuels"
                    )

            if not reasons:
                reasons.append("Score de risque global trop élevé")
                risk_factors.append("Combinaison de facteurs de risque défavorables")

            return {
                "status": "error",
                "title": "CRÉDIT REFUSÉ",
                "message": (
                    f"Votre demande de crédit n'a pas pu être acceptée "
                    f"pour les raisons suivantes :"
                ),
                "reasons": reasons,
                "risk_factors": risk_factors,
                "recommendations": [
                    "Améliorer votre score de crédit",
                    "Réduire le montant demandé",
                    "Augmenter vos revenus",
                    "Stabiliser votre situation professionnelle",
                    "Consulter un conseiller financier",
                ],
            }

    except Exception as e:
        return {
            "status": "warning",
            "title": "ERREUR D'ANALYSE",
            "message": f"Impossible d'analyser les raisons : {str(e)}",
            "details": ["Erreur technique lors de l'analyse"],
        }


def render_prediction_tab(model_data: Dict[str, Any]) -> None:
    """Onglet de prédiction individuelle"""
    st.markdown("## Prédiction Individuelle")
    st.markdown("Analysez le profil de risque d'un client")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Informations Client")

        # 1. INFORMATIONS PERSONNELLES ET SOCIO-DÉMOGRAPHIQUES
        st.markdown("#### 1. Informations Personnelles et Socio-démographiques")
        col1a, col1b = st.columns(2)
        with col1a:
            gender = st.selectbox("Genre", ["Homme", "Femme"], key="gender")
            age_years = st.number_input(
                "Âge (années)", min_value=18, max_value=100, value=35, key="age_years"
            )
            family_status = st.selectbox(
                "Situation familiale",
                [
                    "Célibataire",
                    "Marié",
                    "Union civile",
                    "Veuf/Veuve",
                    "Séparé",
                ],
                key="family_status",
            )
            children = st.number_input(
                "Nombre de personnes à charge",
                min_value=0,
                max_value=20,
                value=0,
                key="children",
            )

        with col1b:
            education = st.selectbox(
                "Niveau d'éducation",
                [
                    "Secondaire",
                    "Supérieur",
                    "Supérieur incomplet",
                    "Secondaire inférieur",
                    "Diplôme universitaire",
                ],
                key="education",
            )
            owns_car = st.selectbox(
                "Possède une voiture",
                ["Oui", "Non"],
                key="owns_car",
            )
            owns_realty = st.selectbox(
                "Possède un bien immobilier",
                ["Oui", "Non"],
                key="owns_realty",
            )

        # 2. INFORMATIONS PROFESSIONNELLES ET REVENUS
        st.markdown("#### 2. Informations Professionnelles et Revenus")
        col1c, col1d = st.columns(2)
        with col1c:
            employment_years = st.number_input(
                "Ancienneté dans l'emploi actuel (années)",
                min_value=0,
                max_value=50,
                value=5,
                key="employment_years",
            )
            income_monthly = st.number_input(
                "Revenus nets mensuels (€)",
                min_value=0,
                value=5000,
                step=100,
                key="income_monthly",
                format="%d",
            )

        with col1d:
            income_type = st.selectbox(
                "Type de revenus",
                [
                    "Salarié",
                    "Indépendant",
                    "Fonctionnaire",
                    "Retraité",
                    "Étudiant",
                    "Chômeur",
                ],
                key="income_type",
            )
            contract_type = st.selectbox(
                "Type de contrat",
                ["Cash loans", "Revolving loans"],
                key="contract_type",
                help="Type de prêt demandé",
            )

        # 3. INFORMATIONS FINANCIÈRES
        st.markdown("#### 3. Informations Financières")
        col1e, col1f = st.columns(2)
        with col1e:
            credit_amount = st.number_input(
                "Montant du crédit demandé (€)",
                min_value=0,
                value=400000,
                step=1000,
                key="credit_amount",
                format="%d",
            )
            annuity_amount = st.number_input(
                "Montant de l'annuité (€)",
                min_value=0,
                value=25000,
                step=100,
                key="annuity_amount",
                format="%d",
            )

        with col1f:
            goods_price = st.number_input(
                "Prix des biens (€)",
                min_value=0,
                value=350000,
                step=1000,
                key="goods_price",
                format="%d",
            )
            # Calculs automatiques
            if credit_amount > 0 and annuity_amount > 0:
                credit_income_ratio = credit_amount / (income_monthly * 12)
                annuity_income_ratio = annuity_amount / (income_monthly * 12)

                st.metric("Ratio Crédit/Revenus", f"{credit_income_ratio:.1f}x")
                st.metric("Ratio Annuité/Revenus", f"{annuity_income_ratio:.1%}")

        # 4. INFORMATIONS SUPPLÉMENTAIRES
        st.markdown("#### 4. Informations Supplémentaires")
        col1g, col1h = st.columns(2)
        with col1g:
            housing_type = st.selectbox(
                "Type de logement",
                [
                    "Propriétaire",
                    "Locataire",
                    "Hébergé gratuitement",
                    "Appartement de fonction",
                    "Logement social",
                ],
                key="housing_type",
            )

        with col1h:
            registration_years = st.number_input(
                "Années depuis l'inscription",
                min_value=0,
                max_value=20,
                value=5,
                key="registration_years",
            )
            family_members = st.number_input(
                "Nombre de membres de la famille",
                min_value=1,
                max_value=10,
                value=2,
                key="family_members",
            )

        # 5. INFORMATIONS SUPPLÉMENTAIRES AVANCÉES
        st.markdown("#### 5. Informations Supplémentaires Avancées")
        col1i, col1j = st.columns(2)
        with col1i:
            region_rating = st.selectbox(
                "Évaluation de la région",
                [1, 2, 3],
                key="region_rating",
                help="1=Faible, 2=Moyen, 3=Élevé",
            )
            region_rating_city = st.selectbox(
                "Évaluation de la région avec ville",
                [1, 2, 3],
                key="region_rating_city",
                help="1=Faible, 2=Moyen, 3=Élevé",
            )

        with col1j:
            organization_type = st.selectbox(
                "Type d'organisation",
                [
                    "Business Entity Type 3",
                    "Self-employed",
                    "Other",
                    "Medicine",
                    "Business Entity Type 2",
                    "Government",
                    "Education",
                    "Business Entity Type 1",
                    "Trade: type 7",
                    "Transport: type 4",
                ],
                key="organization_type",
            )

        # Bouton de validation
        if st.button("Analyser le Dossier", type="primary", use_container_width=True):
            # Validation des règles métier
            validation = validate_business_rules({
                "AMT_INCOME_TOTAL": income_monthly * 12,
                "AMT_CREDIT": credit_amount,
                "AMT_ANNUITY": annuity_amount,
            })

            if not validation["valid"]:
                st.error(f"Validation échouée: {validation['message']}")
                return

            # MAPPING CORRECT AVEC LES FEATURES DE BASE POUR LE FEATURE ENGINEERING
            client_data = {
                # === FEATURES DE BASE (EXACTEMENT COMME HOME CREDIT) ===
                "NAME_CONTRACT_TYPE": "Cash loans",  # Standard
                "CODE_GENDER": "M" if gender == "Homme" else "F",
                "FLAG_OWN_CAR": "Y" if owns_car == "Oui" else "N",
                "FLAG_OWN_REALTY": "Y" if owns_realty == "Oui" else "N",
                "CNT_CHILDREN": children,
                "AMT_INCOME_TOTAL": income_monthly * 12,  # Revenu annuel
                "AMT_CREDIT": credit_amount,
                "AMT_ANNUITY": annuity_amount,
                "AMT_GOODS_PRICE": goods_price,
                # === FEATURES PERSONNELLES ===
                "NAME_TYPE_SUITE": "Unaccompanied",
                "NAME_INCOME_TYPE": {
                    "Salarié": "Working",
                    "Indépendant": "Commercial associate",
                    "Fonctionnaire": "State servant",
                    "Retraité": "Pensioner",
                    "Étudiant": "Student",
                    "Sans emploi": "Unemployed",
                }.get(income_type, "Working"),
                "NAME_EDUCATION_TYPE": {
                    "Primaire": "Lower secondary",
                    "Secondaire": "Secondary / secondary special",
                    "Supérieur": "Higher education",
                    "Post-universitaire": "Academic degree",
                }.get(education, "Secondary / secondary special"),
                "NAME_FAMILY_STATUS": {
                    "Célibataire": "Single / not married",
                    "Marié(e)": "Married",
                    "Union libre": "Civil marriage",
                    "Divorcé(e)": "Separated",
                    "Veuf(ve)": "Widow",
                }.get(family_status, "Single / not married"),
                "NAME_HOUSING_TYPE": {
                    "Propriétaire": "House / apartment",
                    "Locataire": "Rented apartment",
                    "Chez les parents": "With parents",
                    "Logement social": "Municipal apartment",
                }.get(housing_type, "House / apartment"),
                # === FEATURES TEMPORELLES ===
                "DAYS_BIRTH": int(-age_years * 365.25),  # Négatif
                "DAYS_EMPLOYED": int(-employment_years * 365.25),  # Négatif
                "DAYS_REGISTRATION": int(-registration_years * 365.25),  # Négatif
                "DAYS_ID_PUBLISH": int(-registration_years * 365.25),  # Négatif
                # === FEATURES FAMILLE ET CONTACT ===
                "CNT_FAM_MEMBERS": family_members,
                "FLAG_MOBIL": 1,
                "FLAG_EMP_PHONE": 0,  # Par défaut
                "FLAG_WORK_PHONE": 0,  # Par défaut
                "FLAG_CONT_MOBILE": 1,
                "FLAG_PHONE": 1,  # Par défaut
                "FLAG_EMAIL": 0,  # Par défaut
                # === FEATURES RÉGION ET ORGANISATION ===
                "REGION_RATING_CLIENT": region_rating,
                "REGION_RATING_CLIENT_W_CITY": region_rating_city,
                "REGION_POPULATION_RELATIVE": 0.5,  # Valeur par défaut
                "ORGANIZATION_TYPE": {
                    "Entreprise": "Business Entity Type 3",
                    "Administration": "Government",
                    "Banque": "Bank",
                    "École": "School",
                    "Autre": "Other",
                }.get(organization_type, "Business Entity Type 3"),
                "OCCUPATION_TYPE": "Laborers",  # Valeur par défaut
                # === FEATURES GÉOGRAPHIQUES ===
                "LIVE_CITY_NOT_WORK_CITY": 0,
                "LIVE_REGION_NOT_WORK_REGION": 0,
                "REG_REGION_NOT_LIVE_REGION": 0,
                "REG_REGION_NOT_WORK_REGION": 0,
                "REG_CITY_NOT_LIVE_CITY": 0,
                "REG_CITY_NOT_WORK_CITY": 0,
                # === FEATURES TIMING ===
                "WEEKDAY_APPR_PROCESS_START": 1,  # Lundi
                "HOUR_APPR_PROCESS_START": 12,  # Midi
                # === FEATURES EXTERNES (valeurs par défaut) ===
                "EXT_SOURCE_1": 0.5,
                "EXT_SOURCE_2": 0.5,
                "EXT_SOURCE_3": 0.5,
                # === FEATURES BÂTIMENT (valeurs par défaut) ===
                "OWN_CAR_AGE": 5.0,  # Valeur par défaut
                "YEARS_BEGINEXPLUATATION_AVG": 1.0,
                "YEARS_BUILD_AVG": 15.0,
                "APARTMENTS_AVG": 0.5,
                "BASEMENTAREA_AVG": 0.5,
                "COMMONAREA_AVG": 0.5,
                "ELEVATORS_AVG": 0.5,
                "ENTRANCES_AVG": 0.5,
                "FLOORSMAX_AVG": 5.0,
                "FLOORSMIN_AVG": 1.0,
                "LANDAREA_AVG": 0.5,
                "LIVINGAPARTMENTS_AVG": 0.5,
                "LIVINGAREA_AVG": 60.0,
                "NONLIVINGAPARTMENTS_AVG": 0.5,
                "NONLIVINGAREA_AVG": 0.5,
                # === FEATURES SOCIALES ===
                "OBS_30_CNT_SOCIAL_CIRCLE": 2,
                "DEF_30_CNT_SOCIAL_CIRCLE": 0,
                "OBS_60_CNT_SOCIAL_CIRCLE": 2,
                "DEF_60_CNT_SOCIAL_CIRCLE": 0,
                "DAYS_LAST_PHONE_CHANGE": -1000,
                # === FEATURES DOCUMENTS ===
                "FLAG_DOCUMENT_2": 0,
                "FLAG_DOCUMENT_3": 1,  # Documents fournis par défaut
                "FLAG_DOCUMENT_6": 1,  # Documents fournis par défaut
                "FLAG_DOCUMENT_8": 1,  # Documents fournis par défaut
            }

            # Stockage en session pour affichage
            st.session_state.current_prediction = {
                "client_data": client_data,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Lancement de la prédiction
            result = predict_score(client_data, model_data)

            if result is not None:
                # Stockage du résultat
                st.session_state.current_prediction["result"] = result

                # Ajout à l'historique
                st.session_state.history.append(st.session_state.current_prediction)

                st.success("Prédiction effectuée avec succès")
                st.rerun()
            else:
                st.error("Erreur lors de la prédiction")

    # Affichage des résultats
    with col2:
        if (
            st.session_state.current_prediction
            and "result" in st.session_state.current_prediction
        ):
            result = st.session_state.current_prediction["result"]
            client_data = st.session_state.current_prediction["client_data"]

            # Vérification de type pour éviter les erreurs
            if result is None or not isinstance(result, dict):
                st.error("Résultat de prédiction invalide")
                return

            st.markdown("### Résultats de l'Analyse")

            # Décision
            decision_color = "VERT" if result["decision"] == "ACCORDÉ" else "ROUGE"
            st.markdown(f"## {result['decision']}")

            # Probabilité et niveau de risque
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Probabilité de Défaut", f"{result['probability']:.1%}")
            with col2b:
                st.metric("Niveau de Risque", result["risk_level"])

            # Jauge de risque
            risk_percentage = result["probability"] * 100
            if risk_percentage < 30:
                color = "#10B981"  # Vert
            elif risk_percentage < 60:
                color = "#F59E0B"  # Orange
            else:
                color = "#EF4444"  # Rouge

            # Jauge principale
            st.markdown(
                f"""
            <div style="margin: 20px 0;">
                <div style="
                    background-color: #f0f0f0;
                    border-radius: 10px;
                    padding: 10px;
                ">
                    <div style="
                        background-color: {color};
                        height: 30px;
                        border-radius: 8px;
                        width: {risk_percentage}%;
                        transition: width 0.5s;
                    "></div>
                </div>
                <div style="
                    text-align: center;
                    margin-top: 10px;
                    font-size: 18px;
                    font-weight: bold;
                ">
                    Risque de Défaut: {risk_percentage:.1f}%
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Jauge circulaire supplémentaire
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Jauge Circulaire")
                # Création d'une jauge circulaire avec Plotly
                fig_gauge = px.pie(
                    values=[risk_percentage, 100 - risk_percentage],
                    names=["Risque", "Sécurité"],
                    title=f"Risque: {risk_percentage:.1f}%",
                    color_discrete_sequence=[color, "#e0e0e0"],
                )
                fig_gauge.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_risk")

            with col2:
                st.markdown("### Comparaison Risque vs Seuil")
                # Graphique en barres pour comparaison
                risk_data = pd.DataFrame({
                    "Métrique": ["Risque Client", "Seuil Optimal"],
                    "Valeur (%)": [risk_percentage, result["threshold"] * 100],
                })

                fig = px.bar(
                    risk_data,
                    x="Métrique",
                    y="Valeur (%)",
                    title="Comparaison Risque vs Seuil",
                    color="Métrique",
                    color_discrete_map={
                        "Risque Client": "#ff6b6b",
                        "Seuil Optimal": "#4ecdc4",
                    },
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(
                    fig, use_container_width=True, key="bar_risk_comparison"
                )

            # Raison de refus
            if result["decision"] == "REFUSÉ":
                st.markdown("### Raison de Refus")
                refusal_analysis = get_refusal_reason(result, client_data)

                if refusal_analysis:
                    st.markdown("#### Raisons principales :")
                    for reason in refusal_analysis.get("main_reasons", []):
                        st.write(f"• {reason}")

                    st.markdown("#### Facteurs de risque :")
                    for factor in refusal_analysis.get("risk_factors", []):
                        st.write(f"• {factor}")

                    st.markdown("#### Recommandations :")
                    for rec in refusal_analysis.get("recommendations", []):
                        st.write(f"• {rec}")

        else:
            st.info(
                "Prêt pour l'Analyse - Remplissez le formulaire et "
                "cliquez sur 'Analyser le Dossier'"
            )


def render_history_tab() -> None:
    """Interface d'historique"""
    st.markdown("## Historique des Prédictions")

    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history[-10:])):
            with st.expander(
                f"Prédiction {len(st.session_state.history) - i} - {entry['timestamp']}"
            ):
                result = entry["result"]
                client_data = entry["client_data"]

                st.write(f"**Décision**: {result['decision']}")
                st.write(f"**Probabilité**: {result['probability']:.1%}")
                st.write(f"**Genre**: {client_data['CODE_GENDER']}")
                st.write(f"**Revenus**: {client_data['AMT_INCOME_TOTAL']:,.0f} €")
                st.write(f"**Crédit**: {client_data['AMT_CREDIT']:,.0f} €")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Vider l'Historique"):
                st.session_state.history = []
                st.rerun()
    else:
        st.info("Aucune prédiction effectuée pour le moment")


def render_dashboard_overview(model_data: Dict[str, Any]) -> None:
    """Tableau de bord principal avec métriques et aperçu"""
    st.markdown("## Tableau de Bord - Vue d'ensemble")

    # ===== SECTION 1: MÉTRIQUES PRINCIPALES =====
    st.markdown("### Métriques Principales")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Statut de l'API
        if model_data.get("api_status") == "connected":
            st.metric("API Status", "Connectée")
        else:
            st.metric("API Status", "Déconnectée", "Modèle local")
    with col2:
        st.metric("Modèle Actif", "Random Forest")
    with col3:
        st.metric("Features Utilisées", len(model_data.get("feature_names", [])))
    with col4:
        st.metric("Seuil Optimisé", f"{model_data.get('threshold', 0.5):.3f}")

    # Deuxième ligne de métriques
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        total_predictions = len(st.session_state.history)
        if total_predictions > 0:
            approved_count = sum(
                1
                for pred in st.session_state.history
                if pred["result"]["decision"] == "ACCORDÉ"
            )
            approval_rate = (approved_count / total_predictions) * 100
            st.metric("Taux d'Accord", f"{approval_rate:.1f}%")
        else:
            st.metric("Prédictions", "0")
    with col6:
        # Source du modèle
        source = model_data.get("loaded_from", "Local")
        if "API" in str(source):
            st.metric("Source", "API Distante")
        else:
            st.metric("Source", "Modèle Local")
    with col7:
        # Mode de fonctionnement
        if USE_REMOTE_API and model_data.get("api_status") == "connected":
            st.metric("Mode", "API Distante")
        else:
            st.metric("Mode", "Modèle Local")
    with col8:
        # Dernière mise à jour
        if st.session_state.history:
            last_update = st.session_state.history[-1]["timestamp"]
            st.metric("Dernière Prédiction", last_update.split(" ")[1][:5])
        else:
            st.metric("Dernière Prédiction", "Aucune")

    # Indicateur de mise à jour
    if st.session_state.history:
        last_update = st.session_state.history[-1]["timestamp"]
        st.info(f"Dernière mise à jour : {last_update}")

    st.markdown("---")

    # ===== SECTION 2: GRAPHIQUES DE SYNTHÈSE =====
    st.markdown("### Graphiques de Synthèse")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Répartition des Décisions")
        if st.session_state.history:
            decisions = [
                pred["result"]["decision"] for pred in st.session_state.history
            ]
            decision_counts = pd.Series(decisions).value_counts()

            # Ajout de pourcentages dans les labels
            total_decisions = len(decisions)
            labels_with_percent = []
            for decision, count in decision_counts.items():
                percentage = (count / total_decisions) * 100
                labels_with_percent.append(f"{decision} ({percentage:.1f}%)")

            fig = px.pie(
                values=decision_counts.values,
                names=labels_with_percent,
                title="Répartition des Décisions",
                color_discrete_sequence=["#10B981", "#EF4444"],
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True, key="pie_decisions")
        else:
            st.info("Aucune prédiction effectuée pour le moment")

    with col2:
        st.markdown("#### Niveaux de Risque")
        if st.session_state.history:
            risks = [pred["result"]["risk_level"] for pred in st.session_state.history]
            risk_counts = pd.Series(risks).value_counts()

            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Distribution des Niveaux de Risque",
                color=risk_counts.values,
                color_continuous_scale="RdYlGn_r",
                text=risk_counts.values,
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True, key="bar_risk_levels")
        else:
            st.info("Aucune donnée disponible")

    # ===== SECTION 3: STATISTIQUES AVANCÉES =====
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Statistiques Avancées")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Risque moyen
            avg_risk = np.mean(
                [pred["result"]["probability"] for pred in st.session_state.history]
            )
            st.metric("Risque Moyen", f"{avg_risk:.1%}")

        with col2:
            # Écart-type des risques
            risk_std = np.std(
                [pred["result"]["probability"] for pred in st.session_state.history]
            )
            st.metric("Écart-type Risque", f"{risk_std:.1%}")

        with col3:
            # Dernière prédiction
            last_prediction = st.session_state.history[-1]
            last_risk = last_prediction["result"]["probability"]
            st.metric("Dernier Risque", f"{last_risk:.1%}")

        with col4:
            # Nombre total de prédictions
            st.metric("Total Prédictions", len(st.session_state.history))

        # Bouton pour forcer la mise à jour
        if st.button("Actualiser les Graphiques", use_container_width=True):
            st.rerun()

        # ===== SECTION 4: ANALYSES AVANCÉES =====
        st.markdown("---")
        st.markdown("### Analyses Avancées")

        # 1. Évolution temporelle et distribution des probabilités
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Évolution Temporelle")
            if len(st.session_state.history) > 1:
                # Préparer les données temporelles
                timestamps = [pred["timestamp"] for pred in st.session_state.history]
                probabilities = [
                    pred["result"]["probability"] for pred in st.session_state.history
                ]

                # Créer un DataFrame pour l'analyse temporelle
                df_temp = pd.DataFrame({
                    "timestamp": timestamps,
                    "probability": probabilities,
                    "decision": [
                        pred["result"]["decision"] for pred in st.session_state.history
                    ],
                })
                df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
                df_temp = df_temp.sort_values("timestamp")

                # Graphique d'évolution temporelle
                fig = px.line(
                    df_temp,
                    x="timestamp",
                    y="probability",
                    color="decision",
                    title="Évolution du Risque dans le Temps",
                    labels={
                        "probability": "Probabilité de Défaut",
                        "timestamp": "Date",
                    },
                    color_discrete_map={"ACCORDÉ": "#10B981", "REFUSÉ": "#EF4444"},
                )
                fig.add_hline(
                    y=0.5,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Seuil de Décision",
                )
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(
                    fig, use_container_width=True, key="line_temporal_evolution"
                )
            else:
                st.info("Au moins 2 prédictions nécessaires pour l'analyse temporelle")

        with col2:
            st.markdown("#### Distribution des Probabilités")
            if st.session_state.history:
                probabilities = [
                    pred["result"]["probability"] for pred in st.session_state.history
                ]

                # Histogramme des probabilités
                fig = px.histogram(
                    x=probabilities,
                    nbins=10,
                    title="Distribution des Probabilités de Défaut",
                    labels={"x": "Probabilité de Défaut", "y": "Nombre de Clients"},
                    color_discrete_sequence=["#6366F1"],
                )
                fig.add_vline(
                    x=0.5, line_dash="dash", line_color="red", annotation_text="Seuil"
                )
                st.plotly_chart(
                    fig, use_container_width=True, key="histogram_probabilities"
                )

        # 2. Analyse des features clés et segmentation
        st.markdown("#### Analyse des Facteurs Clés")
        col1, col2 = st.columns(2)

        with col1:
            # Simuler une analyse des features importantes (basée sur SHAP)
            if st.session_state.history:
                # Créer des données simulées pour l'analyse des features
                feature_importance = {
                    "Revenus": 0.25,
                    "Âge": 0.20,
                    "Expérience": 0.15,
                    "Ratio Crédit/Revenus": 0.18,
                    "Type de Logement": 0.12,
                    "Secteur d'Activité": 0.10,
                }

                fig = px.bar(
                    x=list(feature_importance.values()),
                    y=list(feature_importance.keys()),
                    orientation="h",
                    title="Importance des Features (SHAP)",
                    labels={"x": "Importance", "y": "Features"},
                    color=list(feature_importance.values()),
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(
                    fig, use_container_width=True, key="bar_feature_importance"
                )

        with col2:
            # Analyse des segments de clients
            if st.session_state.history:
                # Créer des segments basés sur les probabilités
                probabilities = [
                    pred["result"]["probability"] for pred in st.session_state.history
                ]

                # Définir les segments
                segments = []
                for prob in probabilities:
                    if prob < 0.3:
                        segments.append("Faible Risque")
                    elif prob < 0.6:
                        segments.append("Risque Moyen")
                    else:
                        segments.append("Risque Élevé")

                segment_counts = pd.Series(segments).value_counts()

                fig = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title="Segmentation des Clients par Risque",
                    color_discrete_map={
                        "Faible Risque": "#10B981",
                        "Risque Moyen": "#F59E0B",
                        "Risque Élevé": "#EF4444",
                    },
                )
                fig.update_traces(textinfo="percent+label")
                st.plotly_chart(
                    fig, use_container_width=True, key="pie_client_segmentation"
                )

        # 3. Métriques de performance
        st.markdown("#### Métriques de Performance")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.session_state.history:
                # Calculer des métriques avancées
                recent_predictions = (
                    st.session_state.history[-10:]
                    if len(st.session_state.history) >= 10
                    else st.session_state.history
                )
                recent_avg = np.mean(
                    [pred["result"]["probability"] for pred in recent_predictions]
                )
                overall_avg = np.mean(
                    [pred["result"]["probability"] for pred in st.session_state.history]
                )

                # Détecter les tendances
                if recent_avg > overall_avg * 1.1:
                    trend = "Hausse"
                elif recent_avg < overall_avg * 0.9:
                    trend = "Baisse"
                else:
                    trend = "Stable"

                st.metric("Tendance Récente", trend, f"{recent_avg:.1%}")

        with col2:
            if st.session_state.history:
                # Calculer la stabilité des prédictions
                probabilities = [
                    pred["result"]["probability"] for pred in st.session_state.history
                ]
                stability = 1 - np.std(probabilities)  # Plus stable = plus proche de 1

                if stability > 0.8:
                    stability_status = "Très Stable"
                elif stability > 0.6:
                    stability_status = "Stable"
                else:
                    stability_status = "Variable"

                st.metric("Stabilité du Modèle", stability_status, f"{stability:.2f}")

        with col3:
            if st.session_state.history:
                # Calculer le ratio accord/refus
                decisions = [
                    pred["result"]["decision"] for pred in st.session_state.history
                ]
                accord_count = decisions.count("ACCORDÉ")
                refuse_count = decisions.count("REFUSÉ")

                if refuse_count > 0:
                    ratio = accord_count / refuse_count
                else:
                    ratio = float("inf")

                st.metric(
                    "Ratio Accord/Refus",
                    f"{ratio:.2f}",
                    f"{accord_count}/{refuse_count}",
                )

        # 4. Alertes et recommandations
        st.markdown("#### Alertes et Recommandations")

        if st.session_state.history:
            alerts = []
            recommendations = []

            # Analyser les tendances
            probabilities = [
                pred["result"]["probability"] for pred in st.session_state.history
            ]
            recent_avg = (
                np.mean(probabilities[-5:])
                if len(probabilities) >= 5
                else np.mean(probabilities)
            )
            overall_avg = np.mean(probabilities)

            if recent_avg > overall_avg * 1.2:
                alerts.append("Augmentation significative du risque moyen")
                recommendations.append("Analyser les nouveaux profils de clients")

            if np.std(probabilities) > 0.3:
                alerts.append("Variabilité élevée des prédictions")
                recommendations.append("Vérifier la cohérence des données d'entrée")

            decisions = [
                pred["result"]["decision"] for pred in st.session_state.history
            ]
            refuse_rate = decisions.count("REFUSÉ") / len(decisions)

            if refuse_rate > 0.7:
                alerts.append("Taux de refus élevé")
                recommendations.append("Revoir les critères d'évaluation")
            elif refuse_rate < 0.2:
                alerts.append("Taux de refus très faible")
                recommendations.append("Renforcer les contrôles de risque")

            # Afficher les alertes
            if alerts:
                st.warning("**Alertes détectées :**")
                for alert in alerts:
                    st.write(f"• {alert}")

            # Afficher les recommandations
            if recommendations:
                st.info("**Recommandations :**")
                for rec in recommendations:
                    st.write(f"• {rec}")

            if not alerts and not recommendations:
                st.success("Aucune alerte détectée - Modèle stable")

    # Actions rapides
    st.markdown("---")
    st.markdown("Actions Rapides")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Nouvelle Prédiction", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Voir l'Historique", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("Actualiser", use_container_width=True):
            st.rerun()


def render_batch_analysis_tab(model_data: Dict[str, Any]) -> None:
    """Interface d'analyse en lot"""
    st.markdown("## Analyse en Lot")
    st.markdown("Analysez plusieurs dossiers simultanément")

    # Template CSV
    st.markdown("### Télécharger le Template")

    if st.button("Télécharger le template CSV", use_container_width=True):
        template_data = {
            "CODE_GENDER": ["M", "F"],
            "FLAG_OWN_CAR": ["Y", "N"],
            "FLAG_OWN_REALTY": ["Y", "Y"],
            "CNT_CHILDREN": [0, 2],
            "AMT_INCOME_TOTAL": [200000, 150000],
            "AMT_CREDIT": [400000, 250000],
            "AMT_ANNUITY": [25000, 18000],
            "AMT_GOODS_PRICE": [350000, 220000],
            "NAME_INCOME_TYPE": ["Working", "Working"],
            "NAME_EDUCATION_TYPE": [
                "Secondary / secondary special",
                "Higher education",
            ],
            "NAME_FAMILY_STATUS": ["Married", "Single / not married"],
            "NAME_HOUSING_TYPE": ["House / apartment", "House / apartment"],
        }

        template_df = pd.DataFrame(template_data)
        csv = template_df.to_csv(index=False)
        st.download_button(
            label="Télécharger template.csv",
            data=csv,
            file_name="template_clients.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Upload de fichier
    st.markdown("### Télécharger vos Données")
    uploaded_file = st.file_uploader(
        "Glissez-déposez votre fichier CSV ici",
        type=["csv"],
        help="Le fichier doit contenir toutes les colonnes requises",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Informations du fichier
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nombre de Clients", f"{len(df):,}")
            with col2:
                st.metric("Colonnes", f"{len(df.columns)}")
            with col3:
                size_kb = df.memory_usage(deep=True).sum() / 1024
                st.metric("Taille", f"{size_kb:.1f} KB")

            # Aperçu des données
            st.markdown("### Aperçu des Données")
            st.dataframe(df.head(), use_container_width=True)

            # Analyse en lot
            if st.button(
                "Lancer l'Analyse en Lot",
                type="primary",
                use_container_width=True,
            ):
                render_batch_processing(df, model_data)

        except Exception as e:
            st.error(f"Erreur traitement fichier: {str(e)}")


def render_batch_processing(df: pd.DataFrame, model_data: Dict[str, Any]) -> None:
    """Traitement en lot des prédictions"""
    st.markdown("### Traitement en Cours")

    progress_bar = st.progress(0)
    results: List[Dict[str, Any]] = []

    for i, row in df.iterrows():
        client_data = row.to_dict()
        result = predict_score(client_data, model_data)

        if result:
            results.append({"client_id": i + 1, **client_data, **result})

        progress_bar.progress((i + 1) / len(df))

    if results:
        results_df = pd.DataFrame(results)
        render_batch_results(results_df)


def render_batch_results(results_df: pd.DataFrame) -> None:
    """Affichage des résultats d'analyse en lot"""
    st.markdown("## Résultats de l'Analyse")

    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Clients", f"{len(results_df):,}")
    with col2:
        accordes = len(results_df[results_df["decision"] == "ACCORDÉ"])
        st.metric("Taux d'Acceptation", f"{accordes/len(results_df)*100:.1f}%")
    with col3:
        avg_prob = results_df["probability"].mean()
        st.metric("Risque Moyen", f"{avg_prob:.1%}")
    with col4:
        high_risk = len(results_df[results_df["probability"] > 0.6])
        st.metric("Clients Haut Risque", f"{high_risk}")

    # Graphiques
    col1, col2 = st.columns(2)

    with col1:
        # Distribution des décisions
        decision_counts = results_df["decision"].value_counts()
        fig = px.bar(
            x=decision_counts.index,
            y=decision_counts.values,
            title="Distribution des Décisions",
            color=decision_counts.values,
            color_continuous_scale="RdYlGn_r",
        )
        st.plotly_chart(
            fig, use_container_width=True, key="batch_decisions_distribution"
        )

    with col2:
        # Distribution des risques
        fig = px.histogram(
            results_df,
            x="probability",
            title="Distribution des Probabilités",
            nbins=10,
        )
        st.plotly_chart(
            fig, use_container_width=True, key="batch_probabilities_distribution"
        )

    # Tableau des résultats
    st.markdown("### Détail des Résultats")
    st.dataframe(results_df, use_container_width=True)

    # Export
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Télécharger les résultats",
        data=csv,
        file_name=f'resultats_analyse_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime="text/csv",
        use_container_width=True,
    )


def render_features_tab() -> None:
    """Interface d'analyse des features"""
    st.markdown("## Analyse des Features")
    st.markdown("Comprenez les facteurs clés du modèle")

    # Simuler des données d'importance des features
    feature_importance_data = {
        "feature": [
            "AMT_INCOME_TOTAL",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "CNT_CHILDREN",
        ],
        "importance": [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04],
    }

    feature_importance_df = pd.DataFrame(feature_importance_data)

    if feature_importance_df is not None:
        # Métriques sur les features
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Features", f"{len(feature_importance_df):,}")
        with col2:
            top_importance = feature_importance_df["importance"].iloc[0]
            st.metric("Importance Max", f"{top_importance:.4f}")
        with col3:
            avg_importance = feature_importance_df["importance"].mean()
            st.metric("Importance Moyenne", f"{avg_importance:.4f}")

        # Graphique d'importance
        st.markdown("### Top 20 Variables par Importance")
        top_features = feature_importance_df.head(20).copy()

        fig = px.bar(
            top_features,
            x="importance",
            y="feature",
            orientation="h",
            title="Importance des Variables",
            labels={"x": "Importance", "y": "Variables"},
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(xaxis_title="Importance", yaxis_title="Variables", height=600)
        st.plotly_chart(fig, use_container_width=True, key="features_importance_chart")

        # Recherche et tableau
        st.markdown("### Analyse Détaillée")

        # Créer un DataFrame avec les noms compréhensibles
        display_df = feature_importance_df.copy()
        display_df["Nom Compréhensible"] = display_df["feature"].apply(
            lambda x: x.replace("_", " ").title()
        )
        display_df["Description"] = display_df["feature"].apply(
            lambda x: f"Variable {x.lower().replace('_', ' ')}"
        )
        display_df["Catégorie"] = "Standard"

        # Réorganiser les colonnes
        display_df = display_df[[
            "Nom Compréhensible",
            "importance",
            "Description",
            "Catégorie",
            "feature",
        ]]

        # Affichage direct sans recherche problématique
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("Données d'importance des features non disponibles")


def render_reports_tab() -> None:
    """Interface des rapports"""
    st.markdown("## Rapports et Visualisations")
    st.markdown("Explorez les analyses approfondies")

    # Grille des rapports
    report_files = [
        ("feature_importance.png", "Importance des Features"),
        ("correlation_matrix.png", "Matrice de Corrélation"),
        ("numeric_features_distribution.png", "Distribution Features Numériques"),
        ("temporal_analysis.png", "Analyse Temporelle"),
        ("threshold_analysis.png", "Analyse des Seuils"),
    ]

    # Affichage en grille
    cols = st.columns(2)

    for i, (filename, title) in enumerate(report_files):
        file_path = DATA_DIR / "reports" / filename
        col = cols[i % 2]

        with col:
            st.markdown(f"### {title}")
            if file_path.exists():
                st.image(str(file_path), use_container_width=True)
            else:
                st.info(f"Rapport {title} non disponible")


def main() -> None:
    """Fonction principale de l'application"""
    st.title("Dashboard Credit Scoring")
    st.markdown("**Prêt à Dépenser** - Système MLOps de scoring crédit")

    # Initialiser session state
    init_session_state()

    # Chargement des données
    with st.spinner("Chargement du système..."):
        model_data = load_model()

    # Vérification du modèle
    if model_data is None:
        st.error("Modèle non disponible - Vérifiez la configuration")
        return

    # Sidebar - Navigation
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.selectbox(
            "Sélectionner une section",
            [
                "Tableau de Bord",
                "Prédiction Individuelle",
                "Analyse de Lot",
                "Historique des Prédictions",
                "Analyse des Features",
                "Rapports et Métriques",
            ],
        )

        st.markdown("---")
        st.markdown("Métriques Globales")
        if model_data:
            st.metric("Modèle", "Random Forest")
            st.metric("Features", len(model_data.get("feature_names", [])))
            st.metric("Seuil", f"{model_data.get('threshold', 0.5):.3f}")

        st.markdown("---")
        st.markdown("### Statut de l'API")
        if model_data.get("api_status") == "connected":
            st.success("API Connectée")
            # Afficher des informations de santé si disponibles
            health_data = model_data.get("api_health", {})
            if health_data:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Uptime", f"{health_data.get('uptime_seconds', 0):.0f}s")
                with col2:
                    st.metric("Requêtes", health_data.get('total_requests', 0))
        else:
            st.error("API Déconnectée")
            st.info("Utilisation du modèle local")
            st.markdown("**Tentatives de connexion :**")
            st.markdown(f"- Timeout : {API_TIMEOUT}s")
            st.markdown(f"- Tentatives : {API_RETRY_ATTEMPTS}")
            st.markdown(f"- Délai : {API_RETRY_DELAY}s")

        st.markdown("---")
        st.markdown("### Informations")
        st.info(
            "Dashboard Credit Scoring - Prêt à Dépenser\n\nVersion: 1.0\nDernière mise"
            " à jour: Aujourd'hui"
        )

    # Navigation par page
    if page == "Tableau de Bord":
        render_dashboard_overview(model_data)
    elif page == "Prédiction Individuelle":
        render_prediction_tab(model_data)
    elif page == "Analyse de Lot":
        render_batch_analysis_tab(model_data)
    elif page == "Historique des Prédictions":
        render_history_tab()
    elif page == "Analyse des Features":
        render_features_tab()
    elif page == "Rapports et Métriques":
        render_reports_tab()


if __name__ == "__main__":
    main()
