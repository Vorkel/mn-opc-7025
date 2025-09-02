"""
Configuration globale pour les tests pytest
"""

import os
import shutil
import sys
import tempfile
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Ajouter les répertoires au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))


@pytest.fixture(scope="session")
def temp_dir() -> Generator[str, None, None]:
    """Fixture pour créer un répertoire temporaire pour les tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def sample_data() -> pd.DataFrame:
    """Fixture pour créer des données de test"""
    import numpy as np
    import pandas as pd

    # Données de test pour le scoring crédit
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame({
        "CODE_GENDER": np.random.choice(["M", "F"], n_samples),
        "AMT_INCOME_TOTAL": np.random.uniform(50000, 300000, n_samples),
        "AMT_CREDIT": np.random.uniform(100000, 500000, n_samples),
        "AMT_ANNUITY": np.random.uniform(5000, 50000, n_samples),
        "AMT_GOODS_PRICE": np.random.uniform(80000, 400000, n_samples),
        "NAME_EDUCATION_TYPE": np.random.choice(
            ["Secondary / secondary special", "Higher education"], n_samples
        ),
        "NAME_FAMILY_STATUS": np.random.choice(
            ["Married", "Single / not married"], n_samples
        ),
        "NAME_HOUSING_TYPE": np.random.choice(
            ["House / apartment", "Rented apartment"], n_samples
        ),
        "DAYS_BIRTH": np.random.uniform(-20000, -8000, n_samples),
        "DAYS_EMPLOYED": np.random.uniform(-10000, -1000, n_samples),
        "DAYS_REGISTRATION": np.random.uniform(-5000, -1000, n_samples),
        "DAYS_ID_PUBLISH": np.random.uniform(-3000, -500, n_samples),
        "FLAG_OWN_CAR": np.random.choice(["Y", "N"], n_samples),
        "FLAG_OWN_REALTY": np.random.choice(["Y", "N"], n_samples),
        "CNT_CHILDREN": np.random.randint(0, 5, n_samples),
        "CNT_FAM_MEMBERS": np.random.randint(1, 8, n_samples),
        "EXT_SOURCE_1": np.random.uniform(0, 1, n_samples),
        "EXT_SOURCE_2": np.random.uniform(0, 1, n_samples),
        "EXT_SOURCE_3": np.random.uniform(0, 1, n_samples),
    })

    return data


@pytest.fixture(scope="session")
def sample_target() -> np.ndarray:
    """Fixture pour créer la variable cible de test"""
    import numpy as np

    np.random.seed(42)
    n_samples = 100

    # Créer une variable cible binaire avec déséquilibre
    target = np.random.binomial(1, 0.3, n_samples)

    return target


@pytest.fixture(scope="session")
def mock_model() -> MagicMock:
    """Fixture pour créer un modèle mock"""
    model = MagicMock()

    # Simuler les prédictions
    def predict_proba(X) -> np.ndarray:  # type: ignore
        import numpy as np

        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.random.uniform(0, 1, (n_samples, 2))

    def predict(X) -> np.ndarray:  # type: ignore
        import numpy as np

        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.random.binomial(1, 0.3, n_samples)

    model.predict_proba = predict_proba
    model.predict = predict
    model.feature_importances_ = np.random.uniform(0, 1, 20)

    return model


@pytest.fixture(scope="session")
def mock_api_client() -> MagicMock:
    """Fixture pour créer un client API mock"""
    from unittest.mock import MagicMock

    client = MagicMock()

    # Simuler les réponses de l'API
    def mock_get(url) -> MagicMock:
        response = MagicMock()
        if url == "/health":
            response.status_code = 200
            response.json.return_value = {"status": "healthy"}
        elif url == "/feature_importance":
            response.status_code = 200
            response.json.return_value = {
                "feature_importance": {"feature_1": 0.5, "feature_2": 0.3}
            }
        else:
            response.status_code = 404
        return response

    def mock_post(url, json=None) -> MagicMock:
        response = MagicMock()
        if url == "/predict":
            response.status_code = 200
            response.json.return_value = {"probability": 0.25, "risk_level": "LOW"}
        elif url == "/batch_predict":
            response.status_code = 200
            response.json.return_value = {
                "predictions": [
                    {"probability": 0.25, "risk_level": "LOW"},
                    {"probability": 0.75, "risk_level": "HIGH"},
                ]
            }
        elif url == "/explain":
            response.status_code = 200
            response.json.return_value = {
                "shap_values": [0.1, 0.2, 0.3],
                "feature_names": ["feature_1", "feature_2", "feature_3"],
            }
        else:
            response.status_code = 404
        return response

    client.get = mock_get
    client.post = mock_post

    return client


@pytest.fixture(autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """Setup automatique de l'environnement de test"""
    # Sauvegarder les variables d'environnement originales
    original_env = os.environ.copy()

    # Définir des variables d'environnement pour les tests
    os.environ["MODEL_PATH"] = "models/test_model.pkl"
    os.environ["LOG_LEVEL"] = "DEBUG"

    yield

    # Restaurer les variables d'environnement
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="function")
def clean_logs() -> Generator[None, None, None]:
    """Fixture pour nettoyer les logs avant chaque test"""
    log_files = ["logs/api.log", "logs/security.log"]

    # Sauvegarder les logs existants
    for log_file in log_files:
        if os.path.exists(log_file):
            backup_file = f"{log_file}.backup"
            shutil.copy2(log_file, backup_file)

    yield

    # Restaurer les logs
    for log_file in log_files:
        backup_file = f"{log_file}.backup"
        if os.path.exists(backup_file):
            shutil.move(backup_file, log_file)


# Configuration pour les tests de performance
def pytest_configure(config) -> None:
    """Configuration pytest pour les marqueurs personnalisés"""
    config.addinivalue_line("markers", "unit: Tests unitaires")
    config.addinivalue_line("markers", "integration: Tests d'intégration")
    config.addinivalue_line("markers", "api: Tests API")
    config.addinivalue_line("markers", "performance: Tests de performance")
    config.addinivalue_line("markers", "slow: Tests lents")


# Gestion des erreurs de test
def pytest_runtest_setup(item) -> None:
    """Setup avant chaque test"""
    # Vérifier les dépendances
    if "api" in item.keywords:
        try:
            import fastapi
        except ImportError:
            pytest.skip("FastAPI non installé")

    if "performance" in item.keywords:
        try:
            import locust
        except ImportError:
            pytest.skip("Locust non installé pour les tests de performance")


def pytest_runtest_teardown(item, nextitem) -> None:
    """Teardown après chaque test"""
    # Nettoyer les fichiers temporaires créés pendant les tests
    temp_files = ["test_threshold.png", "test_drift_report.html", "test_model.pkl"]

    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass
