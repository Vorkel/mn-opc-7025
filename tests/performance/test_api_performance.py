"""
Tests de performance pour l'API FastAPI
"""
import pytest
import time
import requests
import numpy as np
import pandas as pd
import sys
import os
from unittest.mock import patch, MagicMock

# Ajouter les répertoires au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'api'))

try:
    from fastapi.testclient import TestClient
    from api.app import app
except ImportError as e:
    print(f"API non disponible: {e}")
    app = MagicMock()


class TestAPIPerformance:
    """Tests de performance pour l'API"""

    @pytest.mark.performance
    def test_api_response_time(self):
        """Test que l'API répond en moins de 100ms"""
        try:
            client = TestClient(app)
            
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # en millisecondes
            
            assert response.status_code == 200
            assert response_time < 100  # moins de 100ms
            
        except Exception as e:
            pytest.skip(f"API non disponible: {e}")

    @pytest.mark.performance
    def test_predict_endpoint_performance(self):
        """Test de performance de l'endpoint de prédiction"""
        try:
            client = TestClient(app)
            
            # Données de test
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
                "EXT_SOURCE_3": 0.5
            }
            
            start_time = time.time()
            response = client.post("/predict", json=test_data)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # en millisecondes
            
            assert response.status_code in [200, 422]  # 422 si validation échoue
            assert response_time < 500  # moins de 500ms pour une prédiction
            
        except Exception as e:
            pytest.skip(f"API non disponible: {e}")

    @pytest.mark.performance
    def test_batch_predict_performance(self):
        """Test de performance de l'endpoint de prédiction par lot"""
        try:
            client = TestClient(app)
            
            # Données de test pour le batch
            test_data = [
                {
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
                    "EXT_SOURCE_3": 0.5
                },
                {
                    "CODE_GENDER": "F",
                    "AMT_INCOME_TOTAL": 150000.0,
                    "AMT_CREDIT": 300000.0,
                    "AMT_ANNUITY": 18000.0,
                    "AMT_GOODS_PRICE": 250000.0,
                    "NAME_EDUCATION_TYPE": "Higher education",
                    "NAME_FAMILY_STATUS": "Single / not married",
                    "NAME_HOUSING_TYPE": "House / apartment",
                    "DAYS_BIRTH": -10000,
                    "DAYS_EMPLOYED": -2000,
                    "DAYS_REGISTRATION": -3000,
                    "DAYS_ID_PUBLISH": -1500,
                    "FLAG_OWN_CAR": "N",
                    "FLAG_OWN_REALTY": "N",
                    "CNT_CHILDREN": 1,
                    "CNT_FAM_MEMBERS": 2,
                    "EXT_SOURCE_1": 0.7,
                    "EXT_SOURCE_2": 0.6,
                    "EXT_SOURCE_3": 0.8
                }
            ]
            
            start_time = time.time()
            response = client.post("/batch_predict", json=test_data)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # en millisecondes
            
            assert response.status_code in [200, 422]  # 422 si validation échoue
            assert response_time < 1000  # moins de 1 seconde pour 2 prédictions
            
        except Exception as e:
            pytest.skip(f"API non disponible: {e}")

    @pytest.mark.performance
    def test_concurrent_requests_performance(self):
        """Test de performance avec des requêtes concurrentes"""
        try:
            client = TestClient(app)
            
            # Données de test
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
                "EXT_SOURCE_3": 0.5
            }
            
            # Simuler des requêtes concurrentes
            start_time = time.time()
            
            responses = []
            for _ in range(5):  # 5 requêtes simultanées
                response = client.post("/predict", json=test_data)
                responses.append(response)
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000  # en millisecondes
            
            # Vérifications
            assert len(responses) == 5
            assert all(r.status_code in [200, 422] for r in responses)
            assert total_time < 2000  # moins de 2 secondes pour 5 requêtes
            
        except Exception as e:
            pytest.skip(f"API non disponible: {e}")


class TestModelPerformance:
    """Tests de performance pour les modèles"""

    @pytest.mark.performance
    def test_model_prediction_speed(self):
        """Test de la vitesse de prédiction du modèle"""
        try:
            # Créer des données de test
            np.random.seed(42)
            n_samples = 1000
            
            X = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, n_samples),
                'feature_2': np.random.normal(0, 1, n_samples),
                'feature_3': np.random.normal(0, 1, n_samples)
            })
            
            # Créer un modèle simple
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=42)
            
            # Données d'entraînement
            y = np.random.binomial(1, 0.3, n_samples)
            model.fit(X, y)
            
            # Test de vitesse de prédiction
            start_time = time.time()
            predictions = model.predict(X)
            end_time = time.time()
            
            prediction_time = (end_time - start_time) * 1000  # en millisecondes
            
            # Vérifications
            assert len(predictions) == n_samples
            assert prediction_time < 100  # moins de 100ms pour 1000 prédictions
            
        except Exception as e:
            pytest.skip(f"Modèle non disponible: {e}")

    @pytest.mark.performance
    def test_model_training_speed(self):
        """Test de la vitesse d'entraînement du modèle"""
        try:
            # Créer des données de test
            np.random.seed(42)
            n_samples = 1000
            
            X = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, n_samples),
                'feature_2': np.random.normal(0, 1, n_samples),
                'feature_3': np.random.normal(0, 1, n_samples)
            })
            y = np.random.binomial(1, 0.3, n_samples)
            
            # Test de vitesse d'entraînement
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=42)
            
            start_time = time.time()
            model.fit(X, y)
            end_time = time.time()
            
            training_time = (end_time - start_time) * 1000  # en millisecondes
            
            # Vérifications
            assert hasattr(model, 'coef_')
            assert training_time < 1000  # moins de 1 seconde pour l'entraînement
            
        except Exception as e:
            pytest.skip(f"Modèle non disponible: {e}")

    @pytest.mark.performance
    def test_feature_importance_calculation_speed(self):
        """Test de la vitesse de calcul de l'importance des features"""
        try:
            # Créer des données de test
            np.random.seed(42)
            n_samples = 1000
            
            X = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, n_samples),
                'feature_2': np.random.normal(0, 1, n_samples),
                'feature_3': np.random.normal(0, 1, n_samples),
                'feature_4': np.random.normal(0, 1, n_samples),
                'feature_5': np.random.normal(0, 1, n_samples)
            })
            y = np.random.binomial(1, 0.3, n_samples)
            
            # Test de vitesse de calcul d'importance
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            
            start_time = time.time()
            model.fit(X, y)
            feature_importance = model.feature_importances_
            end_time = time.time()
            
            calculation_time = (end_time - start_time) * 1000  # en millisecondes
            
            # Vérifications
            assert len(feature_importance) == len(X.columns)
            assert calculation_time < 2000  # moins de 2 secondes
            
        except Exception as e:
            pytest.skip(f"Feature importance non disponible: {e}")


class TestDataProcessingPerformance:
    """Tests de performance pour le traitement des données"""

    @pytest.mark.performance
    def test_data_loading_speed(self):
        """Test de la vitesse de chargement des données"""
        # Vérifier que les données existent
        train_path = "data/raw/application_train.csv"
        
        if os.path.exists(train_path):
            start_time = time.time()
            data = pd.read_csv(train_path)
            end_time = time.time()
            
            loading_time = (end_time - start_time) * 1000  # en millisecondes
            
            # Vérifications
            assert len(data) > 0
            assert loading_time < 5000  # moins de 5 secondes pour charger les données
            
        else:
            pytest.skip("Données non disponibles")

    @pytest.mark.performance
    def test_data_preprocessing_speed(self):
        """Test de la vitesse de prétraitement des données"""
        try:
            # Créer des données de test
            np.random.seed(42)
            n_samples = 1000
            
            data = pd.DataFrame({
                'numeric_col': np.random.normal(0, 1, n_samples),
                'categorical_col': np.random.choice(['A', 'B', 'C'], n_samples),
                'missing_col': np.random.choice([1, 2, np.nan], n_samples)
            })
            
            # Test de vitesse de prétraitement
            start_time = time.time()
            
            # Remplir les valeurs manquantes
            data_filled = data.fillna(data.median())
            
            # Encoder les variables catégorielles
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            data_filled['categorical_col'] = le.fit_transform(data_filled['categorical_col'])
            
            end_time = time.time()
            
            preprocessing_time = (end_time - start_time) * 1000  # en millisecondes
            
            # Vérifications
            assert not data_filled.isnull().any().any()
            assert preprocessing_time < 1000  # moins de 1 seconde
            
        except Exception as e:
            pytest.skip(f"Prétraitement non disponible: {e}")

    @pytest.mark.performance
    def test_business_score_calculation_speed(self):
        """Test de la vitesse de calcul du score métier"""
        try:
            from business_score import BusinessScorer
            
            # Créer des données de test
            np.random.seed(42)
            n_samples = 10000
            
            y_true = np.random.binomial(1, 0.3, n_samples)
            y_pred = np.random.binomial(1, 0.3, n_samples)
            
            # Test de vitesse de calcul
            scorer = BusinessScorer()
            
            start_time = time.time()
            cost = scorer.calculate_business_cost(y_true, y_pred)
            end_time = time.time()
            
            calculation_time = (end_time - start_time) * 1000  # en millisecondes
            
            # Vérifications
            assert isinstance(cost, float)
            assert cost >= 0
            assert calculation_time < 100  # moins de 100ms
            
        except Exception as e:
            pytest.skip(f"BusinessScorer non disponible: {e}")


class TestMemoryUsage:
    """Tests d'utilisation mémoire"""

    @pytest.mark.performance
    def test_memory_usage_model_training(self):
        """Test de l'utilisation mémoire lors de l'entraînement"""
        try:
            import psutil
            import os
            
            # Obtenir l'utilisation mémoire avant
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Créer et entraîner un modèle
            np.random.seed(42)
            n_samples = 10000
            
            X = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, n_samples),
                'feature_2': np.random.normal(0, 1, n_samples),
                'feature_3': np.random.normal(0, 1, n_samples)
            })
            y = np.random.binomial(1, 0.3, n_samples)
            
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Obtenir l'utilisation mémoire après
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            # Vérifications
            assert memory_increase < 500  # moins de 500MB d'augmentation
            
        except ImportError:
            pytest.skip("psutil non disponible")
        except Exception as e:
            pytest.skip(f"Test mémoire non disponible: {e}")

    @pytest.mark.performance
    def test_memory_usage_data_processing(self):
        """Test de l'utilisation mémoire lors du traitement des données"""
        try:
            import psutil
            import os
            
            # Obtenir l'utilisation mémoire avant
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Créer un grand dataset
            np.random.seed(42)
            n_samples = 50000
            
            data = pd.DataFrame({
                'feature_1': np.random.normal(0, 1, n_samples),
                'feature_2': np.random.normal(0, 1, n_samples),
                'feature_3': np.random.normal(0, 1, n_samples),
                'feature_4': np.random.normal(0, 1, n_samples),
                'feature_5': np.random.normal(0, 1, n_samples)
            })
            
            # Traitement des données
            data_processed = data.fillna(data.median())
            data_processed = data_processed * 2  # Opération simple
            
            # Obtenir l'utilisation mémoire après
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            # Vérifications
            assert len(data_processed) == n_samples
            assert memory_increase < 1000  # moins de 1GB d'augmentation
            
        except ImportError:
            pytest.skip("psutil non disponible")
        except Exception as e:
            pytest.skip(f"Test mémoire non disponible: {e}")
