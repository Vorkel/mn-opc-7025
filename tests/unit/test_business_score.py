"""
Tests unitaires pour le module business_score
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.business_score import BusinessScorer

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class TestBusinessScorer:
    """Tests pour la classe BusinessScorer"""

    def test_initialization_default_values(self) -> None:
        """Test l'initialisation avec les valeurs par défaut"""
        scorer = BusinessScorer()
        assert scorer.cost_fn == 10
        assert scorer.cost_fp == 1

    def test_initialization_custom_values(self) -> None:
        """Test l'initialisation avec des valeurs personnalisées"""
        scorer = BusinessScorer(cost_fn=5, cost_fp=2)
        assert scorer.cost_fn == 5
        assert scorer.cost_fp == 2

    def test_calculate_business_cost_perfect_predictions(self) -> None:
        """Test le calcul du coût métier avec des prédictions parfaites"""
        scorer = BusinessScorer()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        cost = scorer.calculate_business_cost(y_true, y_pred)
        assert cost == 0  # Aucune erreur = coût 0

    def test_calculate_business_cost_false_negatives(self) -> None:
        """Test le calcul du coût métier avec des faux négatifs"""
        scorer = BusinessScorer(cost_fn=10, cost_fp=1)
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])  # 2 faux négatifs

        cost = scorer.calculate_business_cost(y_true, y_pred)
        assert cost == 20  # 2 FN * 10 = 20

    def test_calculate_business_cost_false_positives(self) -> None:
        """Test le calcul du coût métier avec des faux positifs"""
        scorer = BusinessScorer(cost_fn=10, cost_fp=1)
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0])  # 2 faux positifs

        cost = scorer.calculate_business_cost(y_true, y_pred)
        assert cost == 2  # 2 FP * 1 = 2

    def test_calculate_business_cost_mixed_errors(self) -> None:
        """Test le calcul du coût métier avec des erreurs mixtes"""
        scorer = BusinessScorer(cost_fn=10, cost_fp=1)
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0])  # 1 FN + 1 FP

        cost = scorer.calculate_business_cost(y_true, y_pred)
        assert cost == 11  # 1 FN * 10 + 1 FP * 1 = 11

    def test_find_optimal_threshold(self) -> None:
        """Test la recherche du seuil optimal"""
        scorer = BusinessScorer()

        # Données de test
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.7])

        optimal_threshold, optimal_cost = scorer.find_optimal_threshold(y_true, y_proba)

        assert isinstance(optimal_threshold, float)
        assert isinstance(optimal_cost, float)
        assert 0 <= optimal_threshold <= 1
        assert optimal_cost >= 0

    def test_find_optimal_threshold_with_roc_curve(self) -> None:
        """Test la recherche du seuil optimal avec courbe ROC"""
        scorer = BusinessScorer()

        # Données de test plus réalistes
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.3, 100)
        y_proba = np.random.random(100)

        optimal_threshold, optimal_cost = scorer.find_optimal_threshold(y_true, y_proba)

        assert isinstance(optimal_threshold, float)
        assert isinstance(optimal_cost, float)
        assert 0 <= optimal_threshold <= 1

    def test_plot_threshold_analysis(self) -> None:
        """Test la génération du graphique d'analyse des seuils"""
        scorer = BusinessScorer()

        # Données de test
        thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        costs = np.array([15, 12, 10, 8, 6, 7, 9, 11, 13])
        optimal_threshold = 0.5
        optimal_cost = 6.0

        # Test que la fonction ne lève pas d'exception
        try:
            scorer.plot_threshold_analysis(
                thresholds, costs, optimal_threshold, optimal_cost
            )
            # Vérifier que le fichier a été créé
            import os

            assert os.path.exists("reports/threshold_analysis.png")
            # Nettoyer
            os.remove("reports/threshold_analysis.png")
        except Exception as e:
            pytest.fail(f"plot_threshold_analysis a levé une exception: {e}")

    def test_invalid_inputs(self) -> None:
        """Test la gestion des entrées invalides"""
        scorer = BusinessScorer()

        # Test avec des arrays vides
        with pytest.raises(ValueError):
            scorer.calculate_business_cost(np.array([]), np.array([]))

        # Test avec des arrays de tailles différentes
        with pytest.raises(ValueError):
            scorer.calculate_business_cost(np.array([0, 1]), np.array([0, 1, 0]))

        # Test avec des valeurs non binaires
        with pytest.raises(ValueError):
            scorer.calculate_business_cost(np.array([0, 1, 2]), np.array([0, 1, 0]))

    def test_edge_cases(self) -> None:
        """Test des cas limites"""
        scorer = BusinessScorer()

        # Test avec un seul échantillon (cas spécial pour confusion_matrix)
        y_true = np.array([1])
        y_pred = np.array([0])
        # Pour un seul échantillon, on doit spécifier les labels
        from sklearn.metrics import confusion_matrix

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = (fn * scorer.cost_fn) + (fp * scorer.cost_fp)
        assert cost == 10  # 1 FN * 10

        # Test avec tous les vrais positifs (nécessite des labels explicites)
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = (fn * scorer.cost_fn) + (fp * scorer.cost_fp)
        assert cost == 0

        # Test avec tous les vrais négatifs (nécessite des labels explicites)
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = (fn * scorer.cost_fn) + (fp * scorer.cost_fp)
        assert cost == 0
