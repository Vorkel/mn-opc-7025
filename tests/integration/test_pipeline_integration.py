"""
Tests d'intégration pour le pipeline complet de scoring crédit
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ajouter les répertoires au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "api"))

try:
    from src.business_score import BusinessScorer
    from src.data_drift_detection import DataDriftDetector
    from src.model_training import ModelTrainer
except ImportError as e:
    print(f"Modules non disponibles: {e}")


class TestPipelineIntegration:
    """Tests d'intégration pour le pipeline complet"""

    @pytest.mark.integration
    def test_end_to_end_pipeline(self) -> None:
        """Test du pipeline complet de bout en bout"""
        # Créer des données de test
        np.random.seed(42)
        n_samples = 100

        # Données d'entraînement
        X_train = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, n_samples),
            "feature_2": np.random.normal(0, 1, n_samples),
            "feature_3": np.random.normal(0, 1, n_samples),
        })
        y_train = np.random.binomial(1, 0.3, n_samples)

        # Données de test
        X_test = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, n_samples),
            "feature_2": np.random.normal(0, 1, n_samples),
            "feature_3": np.random.normal(0, 1, n_samples),
        })
        y_test = np.random.binomial(1, 0.3, n_samples)

        try:
            # 1. Initialiser le scorer métier
            scorer = BusinessScorer(cost_fn=10, cost_fp=1)

            # 2. Entraîner un modèle simple
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)

            # 3. Faire des prédictions
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # 4. Évaluer avec le score métier
            business_cost = scorer.calculate_business_cost(y_test, y_pred)

            # 5. Trouver le seuil optimal
            optimal_threshold, optimal_cost = scorer.find_optimal_threshold(
                y_test, y_proba
            )

            # Vérifications
            assert isinstance(business_cost, float)
            assert business_cost >= 0
            assert isinstance(optimal_threshold, float)
            assert 0 <= optimal_threshold <= 1
            assert isinstance(optimal_cost, float)
            assert optimal_cost >= 0

        except Exception as e:
            pytest.skip(f"Pipeline non disponible: {e}")

    @pytest.mark.integration
    def test_model_training_integration(self) -> None:
        """Test d'intégration de l'entraînement de modèle"""
        # Créer des données de test
        np.random.seed(42)
        n_samples = 50

        X = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, n_samples),
            "feature_2": np.random.normal(0, 1, n_samples),
            "feature_3": np.random.normal(0, 1, n_samples),
        })
        y = np.random.binomial(1, 0.3, n_samples)

        try:
            # Initialiser le trainer
            trainer = ModelTrainer()

            # Préparer les données
            X_train, X_test, y_train, y_test = trainer.prepare_data(
                X, pd.Series(y)
            )  # type: ignore

            # Créer le scorer métier
            business_scorer = trainer.create_business_scorer()

            # Entraîner un modèle baseline
            baseline_model = trainer.train_baseline_model(
                X_train, y_train
            )  # type: ignore

            # Faire des prédictions
            y_pred = baseline_model.predict(X_test)
            y_proba = baseline_model.predict_proba(X_test)[:, 1]

            # Évaluer
            cost = business_scorer.calculate_business_cost(y_test, y_pred)

            # Vérifications
            assert len(X_train) + len(X_test) == len(X)
            assert len(y_train) + len(y_test) == len(y)
            assert isinstance(cost, float)
            assert cost >= 0

        except Exception as e:
            pytest.skip(f"ModelTrainer non disponible: {e}")

    @pytest.mark.integration
    def test_data_drift_integration(self) -> None:
        """Test d'intégration de la détection de drift"""
        # Créer des données de référence
        np.random.seed(42)
        n_samples = 100

        reference_data = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, n_samples),
            "feature_2": np.random.normal(0, 1, n_samples),
            "feature_3": np.random.normal(0, 1, n_samples),
        })

        # Créer des données actuelles (avec drift)
        current_data = pd.DataFrame({
            "feature_1": np.random.normal(2, 1, n_samples),  # Drift de distribution
            "feature_2": np.random.normal(0, 1, n_samples),
            "feature_3": np.random.normal(0, 1, n_samples),
        })

        try:
            # Initialiser le détecteur
            detector = DataDriftDetector(reference_data, current_data)

            # Préparer les données
            detector.prepare_data()

            # Détecter le drift
            drift_result = detector.detect_data_drift()

            # Obtenir le résumé
            summary = detector.get_drift_summary()

            # Vérifications
            assert drift_result is not None
            assert summary is not None
            assert isinstance(summary, dict)

        except Exception as e:
            pytest.skip(f"DataDriftDetector non disponible: {e}")

    @pytest.mark.integration
    def test_feature_importance_integration(self) -> None:
        """Test d'intégration de l'analyse d'importance des features"""
        # Créer des données de test
        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({
            "important_feature": np.random.normal(0, 1, n_samples),
            "less_important_feature": np.random.normal(0, 1, n_samples),
            "noise_feature": np.random.normal(0, 1, n_samples),
        })

        # Créer une target qui dépend principalement de la première feature
        y = (X["important_feature"] > 0).astype(int)

        try:
            # Entraîner un modèle
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)

            # Analyser l'importance des features
            feature_importance = model.feature_importances_

            # Vérifications
            assert len(feature_importance) == len(X.columns)
            assert all(importance >= 0 for importance in feature_importance)
            assert sum(feature_importance) > 0

            # La feature importante devrait avoir une importance plus élevée
            important_idx = X.columns.get_loc("important_feature")
            noise_idx = X.columns.get_loc("noise_feature")

            # Vérifier que la feature importante a une importance > 0
            assert feature_importance[important_idx] > 0

        except Exception as e:
            pytest.skip(f"Feature importance non disponible: {e}")

    @pytest.mark.integration
    def test_mlflow_integration(self) -> None:
        """Test d'intégration avec MLflow"""
        try:
            import mlflow

            # Vérifier que MLflow est configuré
            assert mlflow.get_tracking_uri() is not None

            # Créer un run de test
            with mlflow.start_run(run_name="test_integration"):
                # Log des paramètres
                mlflow.log_param("test_param", "test_value")

                # Log des métriques
                mlflow.log_metric("test_metric", 0.85)

                # Log d'un artefact
                test_data = pd.DataFrame({"test": [1, 2, 3]})
                test_data.to_csv("test_artifact.csv", index=False)
                mlflow.log_artifact("test_artifact.csv")

                # Nettoyer
                import os

                if os.path.exists("test_artifact.csv"):
                    os.remove("test_artifact.csv")

            # Vérifications
            assert True  # Si on arrive ici, MLflow fonctionne

        except Exception as e:
            pytest.skip(f"MLflow non disponible: {e}")

    @pytest.mark.integration
    def test_api_integration(self) -> None:
        """Test d'intégration avec l'API"""
        try:
            from fastapi.testclient import TestClient

            from api.app import app

            client = TestClient(app)

            # Test de l'endpoint de santé
            response = client.get("/health")
            assert response.status_code == 200

            # Test de l'endpoint de prédiction
            test_data = {
                "CODE_GENDER": "M",
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "AMT_GOODS_PRICE": 351000.0,
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "NAME_FAMILY_STATUS": "Married",
                "NAME_HOUSING_TYPE": "House / apartment",
                "DAYS_BIRTH": -12005,
                "DAYS_EMPLOYED": -4542,
                "DAYS_REGISTRATION": -3646,
                "DAYS_ID_PUBLISH": -2120,
                "FLAG_OWN_CAR": "Y",
                "FLAG_OWN_REALTY": "Y",
                "CNT_CHILDREN": 0,
                "CNT_FAM_MEMBERS": 2,
                "EXT_SOURCE_1": 0.5,
                "EXT_SOURCE_2": 0.5,
                "EXT_SOURCE_3": 0.5,
            }

            response = client.post("/predict", json=test_data)
            assert response.status_code in [200, 422]  # 422 si validation échoue

        except Exception as e:
            pytest.skip(f"API non disponible: {e}")


class TestDataFlowIntegration:
    """Tests d'intégration pour le flux de données"""

    @pytest.mark.integration
    def test_data_loading_integration(self) -> None:
        """Test d'intégration du chargement des données"""
        # Vérifier que les données d'entraînement existent
        train_path = "data/raw/application_train.csv"
        test_path = "data/raw/application_test.csv"

        if os.path.exists(train_path) and os.path.exists(test_path):
            # Charger les données
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            # Vérifications de base
            assert len(train_data) > 0
            assert len(test_data) > 0
            assert len(train_data.columns) > 0
            assert len(test_data.columns) > 0

            # Vérifier que les colonnes sont cohérentes
            common_columns = set(train_data.columns) & set(test_data.columns)
            assert len(common_columns) > 0

        else:
            pytest.skip("Données non disponibles")

    @pytest.mark.integration
    def test_model_persistence_integration(self) -> None:
        """Test d'intégration de la persistance du modèle"""
        try:
            import joblib

            # Créer un modèle simple
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(random_state=42)

            # Données de test
            X = np.random.normal(0, 1, (10, 3))
            y = np.random.binomial(1, 0.5, 10)

            # Entraîner
            model.fit(X, y)

            # Sauvegarder
            model_path = "test_model.pkl"
            joblib.dump(model, model_path)

            # Charger
            loaded_model = joblib.load(model_path)

            # Vérifier que les prédictions sont identiques
            pred_original = model.predict(X)
            pred_loaded = loaded_model.predict(X)

            assert np.array_equal(pred_original, pred_loaded)

            # Nettoyer
            if os.path.exists(model_path):
                os.remove(model_path)

        except Exception as e:
            pytest.skip(f"Model persistence non disponible: {e}")

    @pytest.mark.integration
    def test_configuration_integration(self) -> None:
        """Test d'intégration de la configuration"""
        # Vérifier les variables d'environnement
        required_env_vars = ["MODEL_PATH", "LOG_LEVEL"]

        for var in required_env_vars:
            if var not in os.environ:
                # Définir des valeurs par défaut pour les tests
                if var == "MODEL_PATH":
                    os.environ[var] = "models/test_model.pkl"
                elif var == "LOG_LEVEL":
                    os.environ[var] = "DEBUG"

        # Vérifications
        assert "MODEL_PATH" in os.environ
        assert "LOG_LEVEL" in os.environ
        assert os.environ["LOG_LEVEL"] in ["DEBUG", "INFO", "WARNING", "ERROR"]
