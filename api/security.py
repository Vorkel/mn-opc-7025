#!/usr/bin/env python3
"""
Module de sécurité pour l'API de scoring crédit
- Authentification API Key
- Rate limiting
- Validation stricte des inputs
- Logs de sécurité
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


def _load_api_keys_from_env() -> Dict[str, Dict[str, Union[str, int]]]:
    """Charge les clés API depuis les variables d'environnement.

    Prend en charge:
    - API_KEYS_JSON: mapping JSON { "key": {"name": str, "rate_limit": int}, ... }
    - API_KEYS: liste CSV de triplets key:name:limit (ex: "k1:Demo:100,k2:Prod:1000")

    Retourne un mapping prêt à l'emploi.
    Fournit un fallback de développement si non défini.
    """
    # JSON explicite prioritaire
    raw_json = os.getenv("API_KEYS_JSON")
    if raw_json:
        try:
            data = json.loads(raw_json)
            # Validation minimale de structure
            valid_json: Dict[str, Dict[str, Union[str, int]]] = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    name = v.get("name", "Unknown")
                    limit = int(v.get("rate_limit", 100))
                    valid_json[k] = {"name": str(name), "rate_limit": limit}
            if valid_json:
                return valid_json
        except Exception:
            pass  # on retombera sur les autres méthodes

    # Format compact CSV
    raw_compact = os.getenv("API_KEYS")
    if raw_compact:
        valid: Dict[str, Dict[str, Union[str, int]]] = {}
        try:
            parts = [p.strip() for p in raw_compact.split(",") if p.strip()]
            for p in parts:
                # key:name:limit
                segs = p.split(":")
                if len(segs) >= 2:
                    key = segs[0]
                    name = segs[1]
                    limit = int(segs[2]) if len(segs) > 2 else 100
                    valid[key] = {"name": name, "rate_limit": limit}
            if valid:
                return valid
        except Exception:
            pass

    # Fallback développement (non production)
    return {
        "demo_key_123": {"name": "Demo Client", "rate_limit": 100},
        "test_key_789": {"name": "Test Client", "rate_limit": 50},
    }


# Configuration sécurité (chargée depuis l'environnement si disponible)
VALID_API_KEYS: Dict[str, Dict[str, Union[str, int]]] = _load_api_keys_from_env()

# Rate limiting storage (en production, utiliser Redis)
rate_limit_storage: Dict[str, int] = {}

# Logger sécurité
security_logger = logging.getLogger("security")


class SecurityValidator:
    """Classe pour la validation et sécurité"""

    @staticmethod
    def validate_api_key(api_key: str) -> Optional[Dict]:
        """Valide une clé API"""
        if api_key in VALID_API_KEYS:
            return VALID_API_KEYS[api_key]
        return None

    @staticmethod
    def check_rate_limit(api_key: str, limit: int, window_seconds: int = 3600) -> bool:
        """Vérifie le rate limiting (par heure par défaut)"""
        current_time = int(time.time())
        window_start = current_time - (current_time % window_seconds)

        # Clé unique pour cette fenêtre
        rate_key = f"{api_key}:{window_start}"

        # Nettoyer les anciennes entrées
        old_keys = [
            k
            for k in rate_limit_storage.keys()
            if int(k.split(":")[1]) < window_start - window_seconds
        ]
        for old_key in old_keys:
            del rate_limit_storage[old_key]

        # Vérifier la limite
        current_count = rate_limit_storage.get(rate_key, 0)
        if current_count >= limit:
            return False

        # Incrémenter le compteur
        rate_limit_storage[rate_key] = current_count + 1
        return True

    @staticmethod
    def validate_credit_data(data: Dict) -> List[str]:
        """Validation stricte des données de crédit"""
        errors = []

        # Champs obligatoires
        required_fields = [
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "CNT_CHILDREN",
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
        ]

        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Champ obligatoire manquant: {field}")

        # Validation des types et valeurs
        validations = {
            "CODE_GENDER": lambda x: x in ["M", "F"],
            "FLAG_OWN_CAR": lambda x: x in ["Y", "N"],
            "FLAG_OWN_REALTY": lambda x: x in ["Y", "N"],
            "CNT_CHILDREN": lambda x: isinstance(x, int) and 0 <= x <= 20,
            "AMT_INCOME_TOTAL": (
                lambda x: isinstance(x, (int, float)) and 0 < x <= 10000000
            ),
            "AMT_CREDIT": lambda x: isinstance(x, (int, float)) and 0 < x <= 50000000,
            "AMT_ANNUITY": lambda x: isinstance(x, (int, float)) and 0 < x <= 5000000,
            "AMT_GOODS_PRICE": (
                lambda x: isinstance(x, (int, float)) and 0 < x <= 50000000
            ),
            "DAYS_BIRTH": lambda x: isinstance(x, int) and -30000 <= x < 0,
            "DAYS_EMPLOYED": lambda x: isinstance(x, int) and -20000 <= x <= 0,
            "FLAG_MOBIL": lambda x: x in [0, 1],
            "FLAG_EMP_PHONE": lambda x: x in [0, 1],
            "FLAG_WORK_PHONE": lambda x: x in [0, 1],
            "FLAG_CONT_MOBILE": lambda x: x in [0, 1],
        }

        for field, validator in validations.items():
            if field in data and data[field] is not None:
                try:
                    if not validator(data[field]):
                        errors.append(f"Valeur invalide pour {field}: {data[field]}")
                except Exception:
                    errors.append(f"Type invalide pour {field}: {data[field]}")

        # Validation logique métier
        if "AMT_CREDIT" in data and "AMT_ANNUITY" in data:
            if data["AMT_ANNUITY"] * 12 > data["AMT_CREDIT"] * 1.5:
                errors.append("Annuité incohérente par rapport au montant du crédit")

        if "AMT_CREDIT" in data and "AMT_GOODS_PRICE" in data:
            if data["AMT_CREDIT"] > data["AMT_GOODS_PRICE"] * 1.2:
                errors.append("Crédit supérieur à 120% du prix des biens")

        return errors

    @staticmethod
    def sanitize_input(data: Dict) -> Dict[str, Union[str, int, float, None]]:
        """Nettoie et sécurise les inputs"""
        cleaned_data: Dict[str, Union[str, int, float, None]] = {}

        for key, value in data.items():
            # Nettoyer les clés
            clean_key = re.sub(r"[^A-Z0-9_]", "", str(key).upper())

            # Nettoyer les valeurs
            if isinstance(value, str):
                # Supprimer les caractères dangereux
                str_value = re.sub(r'[<>"\']', "", value.strip())
                # Limiter la longueur
                clean_value: Union[str, int, float, None] = (
                    str_value[:100] if str_value else ""
                )
            elif isinstance(value, (int, float)):
                clean_value = value
            else:
                clean_value = str(value)[:100] if value is not None else ""

            cleaned_data[clean_key] = clean_value

        return cleaned_data

    @staticmethod
    def detect_suspicious_patterns(data: Dict, request: Request) -> List[str]:
        """Détecte des patterns suspects"""
        warnings = []

        # Détection de valeurs extrêmes
        if "AMT_INCOME_TOTAL" in data and data["AMT_INCOME_TOTAL"] > 5000000:
            warnings.append("Revenu exceptionnellement élevé détecté")

        if "AMT_CREDIT" in data and data["AMT_CREDIT"] > 10000000:
            warnings.append("Montant de crédit exceptionnellement élevé")

        # Détection de patterns d'attaque
        user_agent = request.headers.get("user-agent", "").lower()
        suspicious_agents = ["bot", "crawler", "scanner", "hack"]

        for agent in suspicious_agents:
            if agent in user_agent:
                warnings.append(f"User-Agent suspect détecté: {agent}")

        # Détection de requêtes trop fréquentes depuis la même IP
        client_ip = request.client.host if request.client else "unknown"
        current_time = int(time.time())

        # Compter les requêtes de cette IP dans les 5 dernières minutes
        recent_requests_key = f"ip_requests:{client_ip}:{current_time // 300}"
        current_ip_requests = rate_limit_storage.get(recent_requests_key, 0)

        if current_ip_requests > 50:  # Plus de 50 requêtes en 5 minutes
            warnings.append("Trop de requêtes depuis cette IP")

        rate_limit_storage[recent_requests_key] = current_ip_requests + 1

        return warnings


# Classes de sécurité FastAPI
security_scheme = HTTPBearer()


async def get_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
) -> str:
    """Dépendance FastAPI pour extraire et valider la clé API"""
    api_key = credentials.credentials

    # Validation de la clé
    key_info = SecurityValidator.validate_api_key(api_key)
    if not key_info:
        security_logger.warning(
            f"Tentative d'accès avec clé invalide: {api_key[:8]}..."
        )
        raise HTTPException(status_code=401, detail="Clé API invalide")

    return api_key


async def check_rate_limit_dependency(
    request: Request, api_key: str = Depends(get_api_key)
) -> str:
    """Dépendance FastAPI pour vérifier le rate limiting"""

    key_info = VALID_API_KEYS[api_key]
    limit = int(key_info["rate_limit"])

    if not SecurityValidator.check_rate_limit(api_key, limit):
        security_logger.warning(
            f"Rate limit dépassé pour {str(key_info['name'])} (clé: {api_key[:8]}...)"
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit dépassé. Limite: {limit} requêtes/heure",
        )

    # Log de l'accès autorisé
    security_logger.info(
        f"Accès autorisé: {str(key_info['name'])} depuis"
        f" {request.client.host if request.client else 'unknown'}"
    )

    return api_key


def validate_and_sanitize_input(data: Dict, request: Request) -> Dict:
    """Valide et nettoie les données d'entrée"""

    # Nettoyage
    cleaned_data = SecurityValidator.sanitize_input(data)

    # Validation stricte
    validation_errors = SecurityValidator.validate_credit_data(cleaned_data)
    if validation_errors:
        security_logger.warning(f"Erreurs de validation: {validation_errors}")
        raise HTTPException(
            status_code=400,
            detail={"message": "Données invalides", "errors": validation_errors},
        )

    # Détection de patterns suspects
    warnings = SecurityValidator.detect_suspicious_patterns(cleaned_data, request)
    if warnings:
        security_logger.warning(f"Patterns suspects détectés: {warnings}")
        # En production, on pourrait bloquer ou marquer pour review

    return cleaned_data


# Middleware de sécurité pour logging
class SecurityMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> Any:
        if scope["type"] == "http":
            # Enrichir les logs de sécurité
            headers = dict(scope.get("headers", []))

            # Log des tentatives d'accès
            security_logger.info(
                f"Tentative d'accès: {scope.get('method')} {scope.get('path')} "
                f"depuis {scope.get('client', ['unknown', 0])[0]}"
            )

        await self.app(scope, receive, send)


# Configuration des logs de sécurité
def setup_security_logging() -> None:
    """Configure le logging de sécurité (fichier sous logs/security.log)"""
    # S'assurer que le dossier de logs existe
    logs_dir = Path(__file__).resolve().parents[1] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not any(isinstance(h, logging.FileHandler) for h in security_logger.handlers):
        security_handler = logging.FileHandler(str(logs_dir / "security.log"))
        security_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        security_handler.setFormatter(security_formatter)
        security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.INFO)


# Constantes de sécurité pour l'API
SECURITY_CONSTANTS = {
    "ACCORDE": "ACCORDÉ",
    "REFUSE": "REFUSÉ",
    "RISK_FAIBLE": "FAIBLE",
    "RISK_MOYEN": "MOYEN",
    "RISK_ELEVE": "ÉLEVÉ",
}


def get_decision_constant(decision: str) -> str:
    """Retourne les constantes de décision sécurisées"""
    return SECURITY_CONSTANTS.get(f"DECISION_{decision}", decision)


# Export des fonctions principales
__all__ = [
    "SecurityValidator",
    "get_api_key",
    "check_rate_limit_dependency",
    "validate_and_sanitize_input",
    "SecurityMiddleware",
    "setup_security_logging",
    "SECURITY_CONSTANTS",
]
