# app.py - API FastAPI pour le scoring crédit
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import psutil
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from api.security import (
    SecurityMiddleware,
    check_rate_limit_dependency,
    setup_security_logging,
    validate_and_sanitize_input,
)


# Configuration du logging JSON structuré
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Gestion des données extra pour compatibilité
        extra_data = getattr(record, "extra_data", None)
        if extra_data:
            log_entry.update(extra_data)
        return json.dumps(log_entry)


# Configuration du logger
logger = logging.getLogger("credit_scoring_api")
logger.setLevel(logging.INFO)

# Handler pour fichier avec rotation (unifié sous logs/ à la racine projet)
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
file_handler = RotatingFileHandler(
    str(LOG_DIR / "api.log"), maxBytes=10 * 1024 * 1024, backupCount=5
)
file_handler.setFormatter(JSONFormatter())

# Handler pour console
console_handler = logging.StreamHandler()
console_handler.setFormatter(JSONFormatter())

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# Modèle Pydantic pour les données d'entrée
class CreditRequest(BaseModel):
    """
    Modèle pour les données d'entrée du client
    """

    # Informations personnelles
    CODE_GENDER: str = Field(..., description="Genre du client (M/F)")
    FLAG_OWN_CAR: str = Field(..., description="Possède une voiture (Y/N)")
    FLAG_OWN_REALTY: str = Field(..., description="Possède un bien immobilier (Y/N)")
    CNT_CHILDREN: int = Field(0, description="Nombre d'enfants")
    AMT_INCOME_TOTAL: float = Field(..., description="Revenu total du client")
    AMT_CREDIT: float = Field(..., description="Montant du crédit demandé")
    AMT_ANNUITY: float = Field(..., description="Montant de l'annuité")
    AMT_GOODS_PRICE: float = Field(..., description="Prix des biens")

    # Informations professionnelles
    NAME_TYPE_SUITE: Optional[str] = Field(None, description="Type de suite")
    NAME_INCOME_TYPE: str = Field(..., description="Type de revenu")
    NAME_EDUCATION_TYPE: str = Field(..., description="Niveau d'éducation")
    NAME_FAMILY_STATUS: str = Field(..., description="Statut familial")
    NAME_HOUSING_TYPE: str = Field(..., description="Type de logement")

    # Informations sur l'âge et l'expérience
    DAYS_BIRTH: int = Field(..., description="Âge en jours (négatif)")
    DAYS_EMPLOYED: int = Field(..., description="Jours d'emploi (négatif)")
    DAYS_REGISTRATION: float = Field(..., description="Jours depuis l'enregistrement")
    DAYS_ID_PUBLISH: int = Field(..., description="Jours depuis la publication de l'ID")

    # Informations sur les contacts
    FLAG_MOBIL: int = Field(1, description="Possède un mobile")
    FLAG_EMP_PHONE: int = Field(0, description="Téléphone professionnel")
    FLAG_WORK_PHONE: int = Field(0, description="Téléphone de travail")
    FLAG_CONT_MOBILE: int = Field(0, description="Contact mobile")
    FLAG_PHONE: int = Field(0, description="Téléphone")
    FLAG_EMAIL: int = Field(0, description="Email")

    # Informations sur la famille
    CNT_FAM_MEMBERS: float = Field(..., description="Nombre de membres de la famille")

    # Informations sur la région
    REGION_RATING_CLIENT: int = Field(..., description="Note de la région du client")
    REGION_RATING_CLIENT_W_CITY: int = Field(
        ..., description="Note de la région avec ville"
    )

    # Informations sur l'organisation
    ORGANIZATION_TYPE: str = Field(..., description="Type d'organisation")

    # Informations sur le logement
    YEARS_BEGINEXPLUATATION_AVG: Optional[float] = Field(
        None, description="Années moyennes d'exploitation"
    )
    YEARS_BUILD_AVG: Optional[float] = Field(
        None, description="Années moyennes de construction"
    )

    # Informations sociales
    OWN_CAR_AGE: Optional[float] = Field(None, description="Âge de la voiture")
    FLAG_DOCUMENT_3: int = Field(0, description="Document 3")
    FLAG_DOCUMENT_6: int = Field(0, description="Document 6")
    FLAG_DOCUMENT_8: int = Field(0, description="Document 8")

    class Config:
        json_schema_extra = {
            "example": {
                "CODE_GENDER": "M",
                "FLAG_OWN_CAR": "Y",
                "FLAG_OWN_REALTY": "Y",
                "CNT_CHILDREN": 0,
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "AMT_GOODS_PRICE": 351000.0,
                "NAME_TYPE_SUITE": "Unaccompanied",
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "NAME_FAMILY_STATUS": "Single / not married",
                "NAME_HOUSING_TYPE": "House / apartment",
                "DAYS_BIRTH": -9461,
                "DAYS_EMPLOYED": -637,
                "DAYS_REGISTRATION": -3648.0,
                "DAYS_ID_PUBLISH": -2120,
                "FLAG_MOBIL": 1,
                "FLAG_EMP_PHONE": 1,
                "FLAG_WORK_PHONE": 0,
                "FLAG_CONT_MOBILE": 1,
                "FLAG_PHONE": 1,
                "FLAG_EMAIL": 0,
                "CNT_FAM_MEMBERS": 2.0,
                "REGION_RATING_CLIENT": 2,
                "REGION_RATING_CLIENT_W_CITY": 2,
                "ORGANIZATION_TYPE": "Business Entity Type 3",
                "YEARS_BEGINEXPLUATATION_AVG": 0.9722,
                "YEARS_BUILD_AVG": 0.6192,
                "OWN_CAR_AGE": 12.0,
                "FLAG_DOCUMENT_3": 1,
                "FLAG_DOCUMENT_6": 0,
                "FLAG_DOCUMENT_8": 0,
            }
        }


# Modèle Pydantic pour la réponse
class CreditResponse(BaseModel):
    """
    Modèle pour la réponse de l'API
    """

    client_id: Optional[str] = Field(None, description="ID du client")
    probability: float = Field(..., description="Probabilité de défaut")
    decision: str = Field(..., description="Décision du crédit (ACCORDÉ/REFUSÉ)")
    threshold: float = Field(..., description="Seuil de décision utilisé")
    risk_level: str = Field(..., description="Niveau de risque (FAIBLE/MOYEN/ÉLEVÉ)")
    timestamp: datetime = Field(..., description="Horodatage de la prédiction")


class FeatureImportanceResponse(BaseModel):
    """
    Modèle pour la réponse d'importance des features
    """

    client_id: Optional[str] = Field(None, description="ID du client")
    feature_importance: List[Dict[str, float]] = Field(
        ..., description="Importance des features"
    )
    top_factors: List[Dict[str, Any]] = Field(..., description="Facteurs principaux")


# Variables globales pour le modèle et métriques
model: Optional[Any] = None
threshold: Optional[float] = None
feature_names: Optional[List[str]] = None
scaler: Optional[Any] = None
model_load_time: Optional[float] = None
request_count: int = 0
start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Gestionnaire de cycle de vie pour charger le modèle au démarrage
    """
    global model, threshold, feature_names, scaler, model_load_time

    # Initialisation des logs sécurité + dossier
    os.makedirs("logs", exist_ok=True)
    setup_security_logging()

    load_start = time.time()

    try:
        # Chemin modèle: env MODEL_PATH prioritaire, défaut models/best_credit_model.pkl
        model_path = os.getenv("MODEL_PATH", "models/best_credit_model.pkl")

        # Tenter de charger depuis plusieurs emplacements
        possible_paths = [
            model_path,
            os.path.join("models", os.path.basename(model_path)),
            os.path.join("..", os.path.basename(model_path)),
            os.path.join("..", "models", os.path.basename(model_path)),
        ]

        model_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(
                    f"Chargement du modèle depuis : {path}",
                    extra={"extra_data": {"model_path": path}},
                )
                model_data = joblib.load(path)

                if isinstance(model_data, dict):
                    model = model_data.get("model")
                    threshold = model_data.get("threshold", 0.5)
                    feature_names = model_data.get("feature_names", [])
                    scaler = model_data.get("scaler")
                else:
                    model = model_data
                    threshold = 0.5
                    feature_names = []

                model_loaded = True
                model_load_time = time.time() - load_start

                logger.info(
                    "Modèle chargé avec succès",
                    extra={
                        "extra_data": {
                            "load_time_seconds": model_load_time,
                            "threshold": threshold,
                            "features_count": (
                                len(feature_names) if feature_names else 0
                            ),
                            "model_type": type(model).__name__,
                        }
                    },
                )
                break

        if not model_loaded:
            error_msg = f"Modèle non trouvé dans les chemins: {possible_paths}"
            logger.error(
                error_msg, extra={"extra_data": {"attempted_paths": possible_paths}}
            )
            raise FileNotFoundError(error_msg)

    except Exception as e:
        logger.error(
            f"Erreur lors du chargement du modèle: {e}",
            extra={"extra_data": {"error_type": type(e).__name__}},
        )
        raise

    yield

    # Nettoyage au shutdown
    logger.info(
        "Arrêt de l'API",
        extra={
            "extra_data": {
                "uptime_seconds": time.time() - start_time,
                "total_requests": request_count,
            }
        },
    )


# Création de l'application FastAPI
app = FastAPI(
    title="API de Scoring Crédit",
    description=(
        "API pour calculer la probabilité de défaut d'un client et décider de l'octroi"
        " du crédit"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware sécurité (logging des accès)
app.add_middleware(SecurityMiddleware)

# Configuration CORS pour permettre l'accès depuis Streamlit et autres clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware pour les hosts de confiance (sécurité)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # En production, spécifier les hosts autorisés
)


# Middleware pour logging des requêtes
@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    global request_count
    request_count += 1
    start_time = time.time()

    # Logger la requête entrante
    logger.info(
        f"Requête reçue: {request.method} {request.url.path}",
        extra={
            "extra_data": {
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
                "request_id": request_count,
            }
        },
    )

    response = await call_next(request)

    # Logger la réponse
    process_time = time.time() - start_time
    logger.info(
        f"Réponse envoyée: {response.status_code}",
        extra={
            "extra_data": {
                "status_code": response.status_code,
                "process_time_seconds": round(process_time, 4),
                "request_id": request_count,
            }
        },
    )

    response.headers["X-Process-Time"] = str(process_time)
    return response


def get_model() -> Any:
    """
    Dépendance pour obtenir le modèle
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    return model


def preprocess_input(data: CreditRequest) -> pd.DataFrame:
    """
    Prétraite les données d'entrée pour le modèle en utilisant le module src/

    Args:
        data (CreditRequest): Données d'entrée du client

    Returns:
        pd.DataFrame: Données prétraitées avec features calculées
    """
    try:
        # Convertir en DataFrame
        df = pd.DataFrame([data.dict()])

        # Importer et utiliser le module de feature engineering
        from src.feature_engineering import create_features_complete

        # Appliquer le feature engineering complet
        df_engineered = create_features_complete(df)

        logger.info(f"Feature engineering appliqué: {len(df_engineered.columns)} features créées")
        return df_engineered

    except ImportError as e:
        logger.error(f"Impossible d'importer le module feature_engineering: {e}")
        # Fallback sur le traitement de base
        return _basic_preprocessing(df)
    except Exception as e:
        logger.error(f"Erreur lors du feature engineering: {e}")
        # Fallback sur le traitement de base
        return _basic_preprocessing(df)


def _basic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Traitement de base en cas d'échec du feature engineering avancé
    """
    logger.warning("Utilisation du traitement de base")

    # Encodage simple des variables catégorielles
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = pd.Categorical(df[col]).codes

    # Gestion des valeurs manquantes
    df = df.fillna(0)

    return df


def determine_risk_level(probability: float) -> str:
    """
    Détermine le niveau de risque basé sur la probabilité

    Args:
        probability (float): Probabilité de défaut

    Returns:
        str: Niveau de risque
    """
    if probability < 0.3:
        return "FAIBLE"
    elif probability < 0.6:
        return "MOYEN"
    else:
        return "ÉLEVÉ"


@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Endpoint racine avec informations sur l'API
    """
    return {
        "message": "API de Scoring Crédit",
        "version": "1.0.0",
        "status": "En fonctionnement",
        "model_type": type(model).__name__ if model else "Non chargé",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "feature_importance": "/feature_importance/{client_id}",
            "health": "/health",
            "model_info": "/model_info",
        },
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Endpoint de vérification de santé pour Render.com
    """
    try:
        # Vérification du modèle
        model_status = "loaded" if model is not None else "not_loaded"

        # Vérification de la mémoire système
        memory_info = psutil.virtual_memory()
        memory_status = "healthy" if memory_info.percent < 90 else "warning"

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "mn-opc-7025-api",
            "model_status": model_status,
            "memory_usage_percent": round(memory_info.percent, 2),
            "memory_status": memory_status,
            "uptime_seconds": round(time.time() - start_time, 2),
            "total_requests": request_count
        }
    except Exception as e:
        logger.error(f"Erreur lors du health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "service": "mn-opc-7025-api"
        }


@app.get("/model_info")
async def get_model_info() -> Dict[str, Any]:
    """
    Informations sur le modèle chargé
    """
    return {
        "model_type": type(model).__name__ if model else "Non chargé",
        "threshold": threshold,
        "status": "loaded" if model else "not_loaded",
    }


@app.post("/predict", response_model=CreditResponse)
async def predict_credit(
    request: CreditRequest,
    http_request: Request,
    client_id: Optional[str] = None,
    current_model: Any = Depends(get_model),
    api_key: str = Depends(check_rate_limit_dependency),
) -> CreditResponse:
    """
    Prédiction du scoring crédit pour un client
    """
    try:
        # Validation/sanitation sécurité
        cleaned = validate_and_sanitize_input(request.model_dump(), http_request)
        validated = CreditRequest(**cleaned)

        # Prétraitement des données
        df = preprocess_input(validated)

        # Prédiction
        if hasattr(current_model, "predict_proba"):
            probability = current_model.predict_proba(df)[0, 1]
        else:
            raise HTTPException(
                status_code=500, detail="Modèle ne supporte pas predict_proba"
            )

        # Décision
        decision = "REFUSÉ" if probability >= threshold else "ACCORDÉ"

        # Niveau de risque
        risk_level = determine_risk_level(probability)

        response = CreditResponse(
            client_id=client_id,
            probability=float(probability),
            decision=decision,
            threshold=threshold or 0.5,
            risk_level=risk_level,
            timestamp=datetime.now(),
        )

        logger.info(
            f"Prédiction effectuée - Client: {client_id}, Probabilité:"
            f" {probability:.4f}, Décision: {decision}"
        )

        return response

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/predict_public", response_model=CreditResponse)
async def predict_credit_public(
    request: CreditRequest,
    http_request: Request,
    client_id: Optional[str] = None,
    current_model: Any = Depends(get_model),
) -> CreditResponse:
    """
    Prédiction du scoring crédit pour un client (endpoint public sans authentification)
    """
    try:
        # Validation/sanitation sécurité
        cleaned = validate_and_sanitize_input(request.model_dump(), http_request)
        validated = CreditRequest(**cleaned)

        # Prétraitement des données
        df = preprocess_input(validated)

        # Prédiction
        if hasattr(current_model, "predict_proba"):
            probability = current_model.predict_proba(df)[0, 1]
        else:
            raise HTTPException(
                status_code=500, detail="Modèle ne supporte pas predict_proba"
            )

        # Décision
        decision = "REFUSÉ" if probability >= threshold else "ACCORDÉ"

        # Niveau de risque
        risk_level = determine_risk_level(probability)

        response = CreditResponse(
            client_id=client_id,
            probability=float(probability),
            decision=decision,
            threshold=threshold or 0.5,
            risk_level=risk_level,
            timestamp=datetime.now(),
        )

        logger.info(
            f"Prédiction publique effectuée - Client: {client_id}, Probabilité:"
            f" {probability:.4f}, Décision: {decision}"
        )

        return response

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction publique: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(
    requests: List[CreditRequest],
    http_request: Request,
    current_model: Any = Depends(get_model),
    api_key: str = Depends(check_rate_limit_dependency),
) -> Dict[str, Any]:
    """
    Prédiction en lot pour plusieurs clients
    """
    try:
        results = []

        for i, request in enumerate(requests):
            # Prétraitement
            cleaned = validate_and_sanitize_input(request.model_dump(), http_request)
            validated = CreditRequest(**cleaned)
            df = preprocess_input(validated)

            # Prédiction
            if hasattr(current_model, "predict_proba"):
                probability = current_model.predict_proba(df)[0, 1]
            else:
                raise HTTPException(
                    status_code=500, detail="Modèle ne supporte pas predict_proba"
                )
            decision = "REFUSÉ" if probability >= (threshold or 0.5) else "ACCORDÉ"
            risk_level = determine_risk_level(probability)

            results.append({
                "client_index": i,
                "probability": float(probability),
                "decision": decision,
                "risk_level": risk_level,
                "timestamp": datetime.now().isoformat(),
            })

        logger.info(f"Prédiction en lot effectuée pour {len(requests)} clients")

        return {
            "total_clients": len(requests),
            "results": results,
            "threshold": threshold or 0.5,
            "processed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction en lot: {e}")
        raise HTTPException(
            status_code=500, detail=f"Erreur de prédiction en lot: {str(e)}"
        )


@app.get("/feature_importance/{client_id}")
async def get_feature_importance(
    client_id: str,
    current_model: Any = Depends(get_model),
    api_key: str = Depends(check_rate_limit_dependency),
) -> Union[FeatureImportanceResponse, Dict[str, Any]]:
    """
    Analyse d'importance des features pour un client spécifique
    (Implémentation simplifiée - nécessite SHAP pour une analyse complète)
    """
    try:
        # Pour une implémentation complète, il faudrait:
        # 1. Récupérer les données du client
        # 2. Calculer les valeurs SHAP
        # 3. Retourner l'importance des features

        # Exemple simplifié avec importance globale
        if hasattr(current_model, "feature_importances_"):
            importances = current_model.feature_importances_

            # Créer des noms de features fictifs
            feature_names_list = [f"feature_{i}" for i in range(len(importances))]

            feature_importance: List[Dict[str, Union[str, float]]] = [
                {"feature": str(name), "importance": float(imp)}
                for name, imp in zip(feature_names_list, importances)
            ]

            # Trier par importance
            feature_importance.sort(key=lambda x: float(x["importance"]), reverse=True)

            return FeatureImportanceResponse(
                client_id=client_id,
                feature_importance=feature_importance[:20],  # type: ignore # Top 20
                top_factors=[
                    {
                        "feature": str(item["feature"]),
                        "importance": float(item["importance"]),
                    }
                    for item in feature_importance[:5]  # Top 5
                ],
            )
        else:
            return {
                "message": (
                    "Le modèle ne supporte pas l'analyse d'importance des features"
                ),
                "client_id": client_id,
            }

    except Exception as e:
        logger.error(f"Erreur lors de l'analyse d'importance: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse: {str(e)}")


@app.post("/explain/{client_id}")
async def explain_prediction(
    client_id: str,
    request: CreditRequest,
    http_request: Request,
    current_model: Any = Depends(get_model),
    api_key: str = Depends(check_rate_limit_dependency),
) -> Dict[str, Any]:
    """
    Explication détaillée de la prédiction pour un client
    """
    try:
        # Validation/sanitation
        cleaned = validate_and_sanitize_input(request.model_dump(), http_request)
        validated = CreditRequest(**cleaned)

        # Prétraitement
        df = preprocess_input(validated)

        # Prédiction
        if hasattr(current_model, "predict_proba"):
            probability = current_model.predict_proba(df)[0, 1]
        else:
            raise HTTPException(
                status_code=500, detail="Modèle ne supporte pas predict_proba"
            )
        decision = "REFUSÉ" if probability >= (threshold or 0.5) else "ACCORDÉ"

        # Explication simplifiée (pour une explication complète, utiliser SHAP)
        explanation = {
            "client_id": client_id,
            "probability": float(probability),
            "decision": decision,
            "threshold": threshold or 0.5,
            "explanation": {
                "decision_reason": (
                    f"Probabilité de défaut ({probability:.4f})"
                    f" {'supérieure' if probability >= (threshold or 0.5) else 'inférieure'} au"  # noqa: E501
                    f" seuil ({(threshold or 0.5):.4f})"
                ),
                "risk_factors": [
                    "Analyse complète nécessite l'intégration de SHAP",
                    "Montant du crédit: impact moyen",
                    "Revenus: impact élevé",
                    "Historique: impact moyen",
                ],
                "recommendations": [
                    "Vérifier les documents justificatifs",
                    "Analyser l'historique de crédit",
                    "Évaluer la capacité de remboursement",
                ],
            },
            "timestamp": datetime.now().isoformat(),
        }

        return explanation

    except Exception as e:
        logger.error(f"Erreur lors de l'explication: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur d'explication: {str(e)}")


if __name__ == "__main__":
    # Configuration pour le développement
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
