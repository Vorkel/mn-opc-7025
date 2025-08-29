# business_score.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import mlflow
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BusinessScorer:
    """
    Classe pour calculer le score métier et optimiser le seuil de décision
    """

    def __init__(self, cost_fn: int = 10, cost_fp: int = 1) -> None:
        """
        Initialise le scorer avec les coûts métier

        Args:
            cost_fn (int): Coût d'un faux négatif (mauvais client prédit comme bon)
            cost_fp (int): Coût d'un faux positif (bon client prédit comme mauvais)
        """
        self.cost_fn = cost_fn  # Coût d'un crédit accordé à un mauvais client
        self.cost_fp = cost_fp  # Coût d'un crédit refusé à un bon client

    def calculate_business_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule le coût métier basé sur la matrice de confusion

        Args:
            y_true (np.ndarray): Vraies étiquettes
            y_pred (np.ndarray): Prédictions

        Returns:
            float: Coût métier total
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Coût total = (nombre de FN * coût FN) + (nombre de FP * coût FP)
        total_cost = (fn * self.cost_fn) + (fp * self.cost_fp)

        return float(total_cost)

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_proba: np.ndarray, plot: bool = True
    ) -> Tuple[float, float]:
        """
        Trouve le seuil optimal qui minimise le coût métier

        Args:
            y_true (np.ndarray): Vraies étiquettes
            y_proba (np.ndarray): Probabilités prédites
            plot (bool): Afficher le graphique

        Returns:
            Tuple[float, float]: Seuil optimal et coût optimal
        """
        # Calculer les métriques ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)

        costs = []

        # Tester différents seuils
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            cost = self.calculate_business_cost(y_true, y_pred)
            costs.append(cost)

        costs = np.array(costs)  # type: ignore

        # Trouver le seuil qui minimise le coût
        optimal_idx = np.argmin(costs)
        optimal_threshold = float(thresholds[optimal_idx])
        optimal_cost = float(costs[optimal_idx])

        if plot:
            self.plot_threshold_analysis(
                thresholds, costs, optimal_threshold, optimal_cost  # type: ignore
            )

        return optimal_threshold, optimal_cost

    def plot_threshold_analysis(
        self,
        thresholds: np.ndarray,
        costs: np.ndarray,
        optimal_threshold: float,
        optimal_cost: float,
    ) -> None:
        """
        Affiche l'analyse des seuils

        Args:
            thresholds (np.ndarray): Seuils testés
            costs (np.ndarray): Coûts correspondants
            optimal_threshold (float): Seuil optimal
            optimal_cost (float): Coût optimal
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Graphique des coûts en fonction du seuil
        ax1.plot(thresholds, costs, "b-", linewidth=2)
        ax1.axvline(
            x=optimal_threshold,
            color="r",
            linestyle="--",
            label=f"Seuil optimal: {optimal_threshold:.3f}",
        )
        ax1.axhline(y=optimal_cost, color="r", linestyle="--", alpha=0.7)
        ax1.set_xlabel("Seuil de décision")
        ax1.set_ylabel("Coût métier")
        ax1.set_title("Optimisation du seuil de décision")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Zoom sur la zone optimale
        # Trouver une fenêtre autour du seuil optimal
        window_size = 0.1
        mask = (thresholds >= optimal_threshold - window_size) & (
            thresholds <= optimal_threshold + window_size
        )

        if np.any(mask):
            ax2.plot(thresholds[mask], np.array(costs)[mask], "b-", linewidth=2)
            ax2.axvline(
                x=optimal_threshold,
                color="r",
                linestyle="--",
                label=f"Seuil optimal: {optimal_threshold:.3f}",
            )
            ax2.set_xlabel("Seuil de décision")
            ax2.set_ylabel("Coût métier")
            ax2.set_title("Zoom sur la zone optimale")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Sauvegarder dans reports/
        reports_path = Path("reports")
        reports_path.mkdir(exist_ok=True)
        output_path = reports_path / "threshold_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Graphique sauvegardé: {output_path}")
        plt.close()

    def evaluate_model(
        self, y_true: np.ndarray, y_proba: np.ndarray, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Évalue un modèle avec les métriques métier et techniques

        Args:
            y_true (np.ndarray): Vraies étiquettes
            y_proba (np.ndarray): Probabilités prédites
            threshold (Optional[float]): Seuil de décision (si None, trouve l'optimal)

        Returns:
            Dict[str, Any]: Dictionnaire des métriques
        """
        # Trouver le seuil optimal si pas fourni
        if threshold is None:
            threshold, _ = self.find_optimal_threshold(y_true, y_proba, plot=False)

        # Prédictions avec le seuil optimal
        y_pred = (y_proba >= threshold).astype(int)

        # Calculer les métriques
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Métriques métier
        business_cost = self.calculate_business_cost(y_true, y_pred)

        # Métriques techniques
        auc_score = roc_auc_score(y_true, y_proba)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # F1-score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics = {
            "threshold": float(threshold),
            "business_cost": business_cost,
            "auc_score": float(auc_score),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

        return metrics

    def log_metrics_to_mlflow(self, metrics: Dict[str, Any], model_name: str) -> None:
        """
        Enregistre les métriques dans MLFlow

        Args:
            metrics (Dict[str, Any]): Métriques à logger
            model_name (str): Nom du modèle
        """
        try:
            with mlflow.start_run():
                # Métriques principales
                mlflow.log_metric("business_cost", metrics["business_cost"])
                mlflow.log_metric("auc_score", metrics["auc_score"])
                mlflow.log_metric("accuracy", metrics["accuracy"])
                mlflow.log_metric("precision", metrics["precision"])
                mlflow.log_metric("recall", metrics["recall"])
                mlflow.log_metric("specificity", metrics["specificity"])
                mlflow.log_metric("f1_score", metrics["f1_score"])
                mlflow.log_metric("optimal_threshold", metrics["threshold"])

                # Matrice de confusion
                mlflow.log_metric("true_negatives", metrics["confusion_matrix"]["tn"])
                mlflow.log_metric("false_positives", metrics["confusion_matrix"]["fp"])
                mlflow.log_metric("false_negatives", metrics["confusion_matrix"]["fn"])
                mlflow.log_metric("true_positives", metrics["confusion_matrix"]["tp"])

                # Paramètres du modèle
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("cost_fn", self.cost_fn)
                mlflow.log_param("cost_fp", self.cost_fp)
        except Exception as e:
            logger.warning(f"Échec du logging MLFlow: {e}")

    def print_evaluation_report(self, metrics: Dict[str, Any]) -> None:
        """
        Affiche un rapport d'évaluation formaté

        Args:
            metrics (Dict[str, Any]): Métriques à afficher
        """
        print("=" * 50)
        print("RAPPORT D'ÉVALUATION DU MODÈLE")
        print("=" * 50)

        print(f"Seuil optimal: {metrics['threshold']:.4f}")
        print(f"Coût métier: {metrics['business_cost']:.2f}")
        print(f"AUC Score: {metrics['auc_score']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Précision: {metrics['precision']:.4f}")
        print(f"Rappel: {metrics['recall']:.4f}")
        print(f"Spécificité: {metrics['specificity']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")

        print("\nMatrice de confusion:")
        cm = metrics["confusion_matrix"]
        print(f"TN: {cm['tn']:6d} | FP: {cm['fp']:6d}")
        print(f"FN: {cm['fn']:6d} | TP: {cm['tp']:6d}")

        print("\nInterprétation métier:")
        print(f"- Vrais négatifs (TN): {cm['tn']} bons clients correctement identifiés")
        print(
            f"- Faux positifs (FP): {cm['fp']} bons clients refusés (manque à gagner)"
        )
        print(f"- Faux négatifs (FN): {cm['fn']} mauvais clients acceptés (pertes)")
        print(
            f"- Vrais positifs (TP): {cm['tp']} mauvais clients correctement identifiés"
        )


# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple avec des données simulées
    np.random.seed(42)

    # Simulation de données
    n_samples = 1000
    y_true = np.random.binomial(1, 0.2, n_samples)  # 20% de défauts
    y_proba = np.random.beta(2, 8, n_samples)  # Probabilités simulées

    # Créer le scorer métier
    scorer = BusinessScorer(cost_fn=10, cost_fp=1)

    # Trouver le seuil optimal
    optimal_threshold, optimal_cost = scorer.find_optimal_threshold(y_true, y_proba)

    # Évaluer le modèle
    metrics = scorer.evaluate_model(y_true, y_proba, optimal_threshold)

    # Afficher le rapport
    scorer.print_evaluation_report(metrics)

    print(f"\nSeuil optimal trouvé: {optimal_threshold:.4f}")
    print(f"Coût métier optimal: {optimal_cost:.2f}")
