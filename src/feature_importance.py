# feature_importance.py
# type: ignore
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Import SHAP avec fallback
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP non disponible, certaines fonctionnalités seront limitées")
    SHAP_AVAILABLE = False
    shap = None


class FeatureImportanceAnalyzer:
    """
    Classe pour analyser l'importance des features (globale et locale) avec SHAP
    """

    def __init__(
        self, model_path: Optional[str] = None, model: Optional[Any] = None
    ) -> None:
        """
        Initialise l'analyseur

        Args:
            model_path (Optional[str]): Chemin vers le modèle sauvegardé
            model (Optional[Any]): Modèle déjà chargé
        """
        if model_path:
            self.load_model(model_path)
        elif model:
            self.model = model
        else:
            self.model = None

        self.explainer: Optional[Any] = None
        self.shap_values: Optional[np.ndarray] = None
        self.threshold: float = 0.5

    def load_model(self, model_path: str) -> None:
        """
        Charge un modèle depuis un fichier

        Args:
            model_path (str): Chemin vers le fichier modèle
        """
        try:
            # Chercher le modèle avec chemin absolu
            if not Path(model_path).exists():
                # Essayer chemin relatif depuis la racine du projet
                project_root = Path(__file__).parent.parent
                model_path = str(project_root / "models" / "best_credit_model.pkl")

            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                self.model = model_data["model"]
                self.threshold = model_data.get("threshold", 0.5)
            else:
                self.model = model_data
                self.threshold = 0.5
            logger.info(f"Modèle chargé depuis {model_path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            self.model = None

    def create_explainer(self, X_train: pd.DataFrame, model_type: str = "tree") -> None:
        """
        Crée l'explainer SHAP approprié selon le type de modèle

        Args:
            X_train (pd.DataFrame): Données d'entraînement
            model_type (str): Type de modèle ('tree', 'linear', 'kernel')
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP non disponible")
            return

        if self.model is None:
            logger.error("Aucun modèle chargé!")
            return

        try:
            if model_type == "tree":
                # Pour les modèles basés sur les arbres (RF, LightGBM, XGBoost)
                self.explainer = shap.TreeExplainer(self.model)  # type: ignore
            elif model_type == "linear":
                # Pour les modèles linéaires
                self.explainer = shap.LinearExplainer(
                    self.model, X_train
                )  # type: ignore
            else:
                # Explainer générique (plus lent)
                if self.model is not None and hasattr(self.model, "predict_proba"):
                    self.explainer = shap.KernelExplainer(  # type: ignore
                        self.model.predict_proba, X_train.sample(100)
                    )
                else:
                    raise ValueError("Le modèle ne supporte pas predict_proba")

            logger.info(f"Explainer SHAP créé avec succès ({model_type})")

        except Exception as e:
            logger.error(f"Erreur lors de la création de l'explainer: {e}")
            # Fallback vers KernelExplainer
            try:
                if self.model is not None and hasattr(self.model, "predict_proba"):
                    self.explainer = shap.KernelExplainer(  # type: ignore
                        lambda x: self.model.predict_proba(x)[:, 1],
                        X_train.sample(100),  # type: ignore
                    )
                    logger.info("Fallback vers KernelExplainer")
                else:
                    logger.error("Le modèle ne supporte pas predict_proba")
            except Exception as e2:
                logger.error(f"Erreur avec KernelExplainer: {e2}")

    def calculate_shap_values(
        self, X_test: pd.DataFrame, max_samples: int = 1000
    ) -> None:
        """
        Calcule les valeurs SHAP pour les données de test

        Args:
            X_test (pd.DataFrame): Données de test
            max_samples (int): Nombre maximum d'échantillons à traiter
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP non disponible")
            return

        if self.explainer is None:
            logger.error("Explainer non initialisé!")
            return

        # Limiter le nombre d'échantillons pour l'efficacité
        if len(X_test) > max_samples:
            X_sample = X_test.sample(max_samples, random_state=42)
            logger.info(f"Calcul des valeurs SHAP sur {max_samples} échantillons")
        else:
            X_sample = X_test

        try:
            logger.info("Calcul des valeurs SHAP en cours...")
            self.shap_values = self.explainer.shap_values(X_sample)

            # Pour les modèles de classification binaire, prendre les valeurs pour la
            # classe positive
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]

            self.X_test_sample = X_sample
            logger.info("Valeurs SHAP calculées avec succès!")

        except Exception as e:
            logger.error(f"Erreur lors du calcul des valeurs SHAP: {e}")

    def plot_global_importance(self, max_features: int = 20) -> None:
        """
        Affiche l'importance globale des features

        Args:
            max_features (int): Nombre maximum de features à afficher
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP non disponible")
            return

        if self.shap_values is None:
            logger.error("Valeurs SHAP non calculées!")
            return

        plt.figure(figsize=(12, 8))

        # Summary plot - importance globale
        if shap is not None:
            shap.summary_plot(  # type: ignore
                self.shap_values,
                self.X_test_sample,
                max_display=max_features,
                show=False,
            )

        plt.title("Importance globale des features (SHAP)", fontsize=16)
        plt.tight_layout()

        # Sauvegarder dans reports/
        reports_path = Path("reports")
        reports_path.mkdir(exist_ok=True)
        output_path = reports_path / "shap_global_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Graphique d'importance globale sauvegardé: {output_path}")
        plt.close()

        # Bar plot de l'importance moyenne
        plt.figure(figsize=(12, 8))
        if shap is not None:
            shap.summary_plot(  # type: ignore
                self.shap_values,
                self.X_test_sample,
                plot_type="bar",
                max_display=max_features,
                show=False,
            )
        plt.title("Importance moyenne des features (SHAP)", fontsize=16)
        plt.tight_layout()

        # Sauvegarder dans reports/
        output_path = reports_path / "shap_mean_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Graphique d'importance moyenne sauvegardé: {output_path}")
        plt.close()

    def plot_local_explanation(
        self, client_index: int = 0, max_features: int = 15
    ) -> None:
        """
        Affiche l'explication locale pour un client spécifique

        Args:
            client_index (int): Index du client à analyser
            max_features (int): Nombre de features à afficher
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP non disponible")
            return

        if self.shap_values is None:
            logger.error("Valeurs SHAP non calculées!")
            return

        # Vérifier que l'index est valide
        if client_index >= len(self.X_test_sample):
            logger.error(f"Index invalide! Maximum: {len(self.X_test_sample)-1}")
            return

        # Prédiction pour ce client
        client_data = self.X_test_sample.iloc[client_index : client_index + 1]
        if self.model is not None and hasattr(self.model, "predict_proba"):
            prediction_proba = self.model.predict_proba(client_data)[0, 1]
        else:
            raise ValueError("Le modèle ne supporte pas predict_proba")

        logger.info(f"CLIENT {client_index}:")
        logger.info(f"Probabilité de défaut: {prediction_proba:.4f}")
        logger.info(
            f"Décision: {'REFUSÉ' if prediction_proba >= self.threshold else 'ACCORDÉ'}"
        )

        # Waterfall plot
        plt.figure(figsize=(12, 8))
        if shap is not None and self.explainer is not None:
            try:
                shap.waterfall_plot(  # type: ignore
                    (
                        self.explainer.expected_value[1]  # type: ignore
                        if isinstance(self.explainer.expected_value, np.ndarray)
                        else self.explainer.expected_value
                    ),
                    self.shap_values[client_index],
                    self.X_test_sample.iloc[client_index],
                    max_display=max_features,
                    show=False,
                )
            except AttributeError:
                # Fallback for older SHAP versions
                shap.force_plot(  # type: ignore
                    (
                        self.explainer.expected_value[1]  # type: ignore
                        if isinstance(self.explainer.expected_value, np.ndarray)
                        else self.explainer.expected_value
                    ),
                    self.shap_values[client_index],
                    self.X_test_sample.iloc[client_index],
                    matplotlib=True,
                    show=False,
                )
        plt.title(f"Explication locale - Client {client_index}", fontsize=16)
        plt.tight_layout()

        # Sauvegarder dans reports/
        reports_path = Path("reports")
        reports_path.mkdir(exist_ok=True)
        output_path = reports_path / f"waterfall_client_{client_index}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Waterfall plot sauvegardé: {output_path}")
        plt.close()

        # Force plot (alternative)
        plt.figure(figsize=(15, 6))
        if shap is not None and self.explainer is not None:
            shap.force_plot(  # type: ignore
                (
                    self.explainer.expected_value[1]  # type: ignore
                    if isinstance(self.explainer.expected_value, np.ndarray)
                    else self.explainer.expected_value
                ),
                self.shap_values[client_index],
                self.X_test_sample.iloc[client_index],
                matplotlib=True,
                show=False,
            )
        plt.title(f"Force plot - Client {client_index}", fontsize=16)
        plt.tight_layout()

        # Sauvegarder dans reports/
        output_path = reports_path / f"force_plot_client_{client_index}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Force plot sauvegardé: {output_path}")
        plt.close()

    def analyze_feature_interactions(self, feature1: str, feature2: str) -> None:
        """
        Analyse les interactions entre deux features

        Args:
            feature1 (str): Nom de la première feature
            feature2 (str): Nom de la deuxième feature
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP non disponible")
            return

        if self.shap_values is None:
            logger.error("Valeurs SHAP non calculées!")
            return

        # Vérifier que les features existent
        if (
            feature1 not in self.X_test_sample.columns
            or feature2 not in self.X_test_sample.columns
        ):
            logger.error("Une ou plusieurs features n'existent pas!")
            return

        plt.figure(figsize=(12, 8))

        # Dependence plot
        if shap is not None:
            shap.dependence_plot(  # type: ignore
                feature1,
                self.shap_values,
                self.X_test_sample,
                interaction_index=feature2,
                show=False,
            )

        plt.title(f"Interaction entre {feature1} et {feature2}", fontsize=16)
        plt.tight_layout()
        plt.show()

    def generate_explanation_report(
        self, client_index: int, save_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Génère un rapport d'explication pour un client

        Args:
            client_index (int): Index du client
            save_path (Optional[str]): Chemin pour sauvegarder le rapport

        Returns:
            Optional[str]: Contenu du rapport ou None
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP non disponible")
            return None

        if self.shap_values is None:
            logger.error("Valeurs SHAP non calculées!")
            return None

        # Données du client
        client_data = self.X_test_sample.iloc[client_index]
        client_shap = self.shap_values[client_index]
        if self.model is not None and hasattr(self.model, "predict_proba"):
            prediction_proba = self.model.predict_proba(
                client_data.values.reshape(1, -1)
            )[0, 1]
        else:
            raise ValueError("Le modèle ne supporte pas predict_proba")

        # Créer le rapport
        report = f"""
RAPPORT D'EXPLICATION DU SCORING CRÉDIT
========================================

Client ID: {client_index}
Probabilité de défaut: {prediction_proba:.4f}
Décision: {'REFUSÉ' if prediction_proba >= self.threshold else 'ACCORDÉ'}
Seuil de décision: {self.threshold:.4f}

FACTEURS PRINCIPAUX (TOP 10):
"""

        # Trier les features par importance absolue
        feature_importance = list(
            zip(self.X_test_sample.columns, client_shap, client_data.values)
        )

        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        for i, (feature, shap_val, feature_val) in enumerate(feature_importance[:10]):
            impact = "AUGMENTE" if shap_val > 0 else "DIMINUE"
            report += f"\n{i+1}. {feature}:"
            report += f"\n   Valeur: {feature_val:.4f}"
            report += f"\n   Impact SHAP: {shap_val:.4f} ({impact} le risque)"
            report += "\n"

        report += """
INTERPRÉTATION:
- Les valeurs SHAP positives augmentent la probabilité de défaut
- Les valeurs SHAP négatives diminuent la probabilité de défaut
- Plus la valeur absolue est élevée, plus l'impact est important

RECOMMANDATIONS:
"""

        # Recommandations basées sur les facteurs principaux
        top_negative = [x for x in feature_importance if x[1] < 0][:3]
        top_positive = [x for x in feature_importance if x[1] > 0][:3]

        if top_positive:
            report += "\nFacteurs de risque identifiés:"
            for feature, shap_val, _ in top_positive:
                report += f"\n- {feature} (impact: {shap_val:.4f})"

        if top_negative:
            report += "\nFacteurs protecteurs identifiés:"
            for feature, shap_val, _ in top_negative:
                report += f"\n- {feature} (impact: {shap_val:.4f})"

        logger.info(report)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"\nRapport sauvegardé dans: {save_path}")

        return report

    def batch_analysis(self, n_clients: int = 5) -> None:
        """
        Analyse plusieurs clients de manière automatique

        Args:
            n_clients (int): Nombre de clients à analyser
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP non disponible")
            return

        if self.shap_values is None:
            logger.error("Valeurs SHAP non calculées!")
            return

        logger.info("=" * 60)
        logger.info("ANALYSE BATCH DE PLUSIEURS CLIENTS")
        logger.info("=" * 60)

        # Sélectionner des clients avec différents niveaux de risque
        if self.model is not None and hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(self.X_test_sample)[:, 1]
        else:
            raise ValueError("Le modèle ne supporte pas predict_proba")

        # Indices de clients à analyser
        indices = []

        # Client à très faible risque
        low_risk_idx = np.argmin(probabilities)
        indices.append(low_risk_idx)

        # Client à risque élevé
        high_risk_idx = np.argmax(probabilities)
        indices.append(high_risk_idx)

        # Clients au seuil de décision
        threshold_diff = np.abs(probabilities - self.threshold)
        near_threshold_indices = np.argsort(threshold_diff)[: n_clients - 2]
        indices.extend(near_threshold_indices)

        # Analyser chaque client
        for i, idx in enumerate(indices):
            logger.info(f"\n--- CLIENT {idx} ---")
            prob = probabilities[idx]
            logger.info(f"Probabilité: {prob:.4f}")
            logger.info(
                f"Décision: {'REFUSÉ' if prob >= self.threshold else 'ACCORDÉ'}"
            )

            # Top 5 features pour ce client
            client_shap = self.shap_values[idx]
            feature_importance = list(zip(self.X_test_sample.columns, client_shap))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

            logger.info("Top 5 facteurs:")
            for j, (feature, shap_val) in enumerate(feature_importance[:5]):
                impact = "↑" if shap_val > 0 else "↓"
                logger.info(f"  {j+1}. {feature}: {shap_val:.4f} {impact}")


# Exemple d'utilisation
if __name__ == "__main__":
    if not SHAP_AVAILABLE:
        logger.error("SHAP non disponible. Installez-le avec: pip install shap")
        exit(1)

    # Charger le modèle depuis le bon chemin
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "best_credit_model.pkl"

    analyzer = FeatureImportanceAnalyzer(str(model_path))

    # Vérifier que le modèle est chargé
    if analyzer.model is None:
        logger.error("Impossible de charger le modèle. Arrêt du script.")
        exit(1)

    # Simuler des données (remplacez par vos vraies données)
    np.random.seed(42)
    feature_names = [f"feature_{i}" for i in range(20)]
    X_test = pd.DataFrame(
        # type: ignore
        np.random.randn(500, 20),
        columns=pd.Index(feature_names),
    )

    # Créer l'explainer et calculer les valeurs SHAP
    analyzer.create_explainer(X_test.sample(100), model_type="tree")
    analyzer.calculate_shap_values(X_test)

    # Analyses
    logger.info("1. Importance globale des features")
    analyzer.plot_global_importance()

    logger.info("\n2. Explication locale pour le client 0")
    analyzer.plot_local_explanation(0)

    logger.info("\n3. Rapport d'explication")
    analyzer.generate_explanation_report(0, "client_0_explanation.txt")

    logger.info("\n4. Analyse batch")
    analyzer.batch_analysis()

    logger.info("\n✅ Analyse d'importance terminée!")
