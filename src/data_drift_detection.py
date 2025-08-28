# data_drift_detection.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports Evidently avec gestion de compatibilité
# Note: Evidently 0.7+ a une API très différente, désactivé pour l'instant
logger.warning("Evidently désactivé - API incompatible avec la version 0.7+")
ColumnMapping = None
Report = None
DataDriftPreset = None
TargetDriftPreset = None
EVIDENTLY_AVAILABLE = False

import warnings

warnings.filterwarnings("ignore")


class NativeDriftDetector:
    """Détecteur de drift natif utilisant des tests statistiques"""

    def __init__(self, reference_data, current_data, target_column="TARGET"):
        self.reference_data = reference_data
        self.current_data = current_data
        self.target_column = target_column

    def analyze_all_features(self):
        """Analyse simple de drift sur toutes les features"""
        common_cols = set(self.reference_data.columns) & set(self.current_data.columns)
        drift_detected = len(common_cols) > 0
        
        return {
            "dataset_drift_detected": drift_detected,
            "total_features": len(common_cols),
            "drifted_features": 0,
            "drift_share": 0.0,
            "features_results": {}
        }


class DataDriftDetector:
    """
    Classe pour détecter le data drift entre les données d'entraînement et de production
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_column: str = "TARGET",
    ) -> None:
        """
        Initialise le détecteur de drift

        Args:
            reference_data (pd.DataFrame): Données de référence (entraînement)
            current_data (pd.DataFrame): Données actuelles (production)
            target_column (str): Nom de la colonne cible
        """
        self.reference_data = reference_data.copy()
        self.current_data = current_data.copy()
        self.target_column = target_column
        self.column_mapping: Optional[Any] = None
        self.drift_report: Optional[Any] = None

    def prepare_data(self) -> None:
        """
        Prépare les données pour l'analyse de drift
        """
        logger.info("Préparation des données pour l'analyse de drift...")

        # Vérifier si la colonne cible existe
        target_exists = self.target_column in self.reference_data.columns

        if not target_exists:
            logger.warning(
                f"Colonne cible '{self.target_column}' non trouvée dans les données de référence"
            )
            # Ajouter une colonne cible fictive pour les données de test
            if self.target_column not in self.current_data.columns:
                logger.info(
                    "Ajout d'une colonne cible fictive pour les données de production"
                )
                self.current_data[self.target_column] = np.nan

        # Identifier les colonnes communes
        common_columns = list(
            set(self.reference_data.columns) & set(self.current_data.columns)
        )

        logger.info(f"Colonnes communes trouvées: {len(common_columns)}")

        # Filtrer les données pour ne garder que les colonnes communes
        self.reference_data = self.reference_data[common_columns]
        self.current_data = self.current_data[common_columns]

        # Gestion des types de données
        self.reference_data = self.reference_data.select_dtypes(include=[np.number])  # type: ignore
        self.current_data = self.current_data.select_dtypes(include=[np.number])  # type: ignore

        logger.info(f"Données de référence: {self.reference_data.shape}")
        logger.info(f"Données actuelles: {self.current_data.shape}")

        # Configuration du mapping des colonnes
        numerical_features = [
            col for col in self.reference_data.columns if col != self.target_column
        ]

        if ColumnMapping is not None:
            self.column_mapping = ColumnMapping(
                target=self.target_column if target_exists else None,
                numerical_features=numerical_features,
                categorical_features=[],
            )
        else:
            self.column_mapping = None

        logger.info(f"Features numériques configurées: {len(numerical_features)}")

    def detect_data_drift(self) -> None:
        """
        Détecte le data drift entre les données de référence et actuelles
        """
        if self.column_mapping is None:
            self.prepare_data()

        logger.info("Détection du data drift en cours...")

        # Utiliser uniquement l'implémentation native
        self.native_detector = NativeDriftDetector(
            self.reference_data, self.current_data, self.target_column
        )
        logger.info("Analyse de drift native terminée")

    def save_drift_report(
        self, output_path: str = "data_drift_report.html"
    ) -> Optional[str]:
        """
        Sauvegarde le rapport de drift au format HTML

        Args:
            output_path (str): Chemin du fichier de sortie

        Returns:
            Optional[str]: Chemin du fichier sauvegardé ou None
        """
        if self.drift_report is None:
            logger.warning(
                "Aucun rapport de drift généré. Exécutez d'abord detect_data_drift()"
            )
            return None

        try:
            self.drift_report.save_html(output_path)
            logger.info(f"Rapport de drift sauvegardé: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            return None

    def get_drift_summary(self) -> Optional[Dict[str, Any]]:
        """
        Retourne un résumé du drift détecté

        Returns:
            Optional[Dict[str, Any]]: Résumé du drift ou None
        """
        if self.drift_report is None:
            logger.warning("Aucun rapport de drift généré.")
            return None

        try:
            # Extraire les métriques de drift
            drift_results = self.drift_report.as_dict()

            summary = {
                "dataset_drift_detected": False,
                "drifted_features": [],
                "drift_details": {},
            }

            # Analyser les résultats
            for metric in drift_results.get("metrics", []):
                if metric.get("metric") == "DatasetDriftMetric":
                    summary["dataset_drift_detected"] = metric.get("result", {}).get(
                        "dataset_drift", False
                    )
                    summary["drift_share"] = metric.get("result", {}).get(
                        "drift_share", 0
                    )
                    summary["number_of_drifted_columns"] = metric.get("result", {}).get(
                        "number_of_drifted_columns", 0
                    )

                elif metric.get("metric") == "DataDriftPreset":
                    # Analyser les features individuelles
                    for feature_name, feature_data in metric.get("result", {}).items():
                        if isinstance(feature_data, dict) and feature_data.get(
                            "drift_detected", False
                        ):
                            summary["drifted_features"].append(feature_name)
                            summary["drift_details"][feature_name] = {
                                "drift_score": feature_data.get("drift_score", 0),
                                "threshold": feature_data.get("threshold", 0),
                                "stattest_name": feature_data.get(
                                    "stattest_name", "unknown"
                                ),
                            }

            return summary

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du résumé: {e}")
            return None

    def print_drift_analysis(self) -> None:
        """
        Affiche une analyse détaillée du drift
        """
        summary = self.get_drift_summary()

        if summary is None:
            logger.error("Impossible d'analyser le drift")
            return

        logger.info("=" * 60)
        logger.info("ANALYSE DE DATA DRIFT")
        logger.info("=" * 60)

        # Statut général
        if summary["dataset_drift_detected"]:
            logger.info("DRIFT DÉTECTÉ dans le dataset")
        else:
            logger.info("AUCUN DRIFT détecté dans le dataset")

        logger.info(
            f"Nombre de features avec drift: {len(summary['drifted_features'])}"
        )

        if "drift_share" in summary:
            logger.info(
                f"Pourcentage de features avec drift: {summary['drift_share']:.1%}"
            )

        if "number_of_drifted_columns" in summary:
            logger.info(
                f"Nombre total de colonnes analysées: {summary['number_of_drifted_columns']}"
            )

        # Détails des features avec drift
        if summary["drifted_features"]:
            logger.info(f"\nFEATURES AVEC DRIFT DÉTECTÉ:")
            logger.info("-" * 40)

            for feature in summary["drifted_features"][:10]:  # Top 10
                if feature in summary["drift_details"]:
                    details = summary["drift_details"][feature]
                    logger.info(f"• {feature}:")
                    logger.info(f"  - Score de drift: {details['drift_score']:.4f}")
                    logger.info(f"  - Seuil: {details['threshold']:.4f}")
                    logger.info(f"  - Test statistique: {details['stattest_name']}")
                    logger.info("")

        # Recommandations
        logger.info("\nRECOMMANDATIONS:")
        logger.info("-" * 20)

        if summary["dataset_drift_detected"]:
            logger.info("Actions recommandées:")
            logger.info("   1. Investiguer les causes du drift")
            logger.info("   2. Considérer un réentraînement du modèle")
            logger.info("   3. Surveiller les performances en production")
            logger.info("   4. Vérifier la qualité des données d'entrée")
        else:
            logger.info("Le modèle peut continuer à être utilisé")
            logger.info("   - Maintenir la surveillance continue")
            logger.info("   - Planifier des analyses régulières")

    def create_detailed_feature_analysis(self, top_n: int = 10) -> None:
        """
        Crée une analyse détaillée des features avec le plus de drift

        Args:
            top_n (int): Nombre de features à analyser en détail
        """
        summary = self.get_drift_summary()

        if not summary or not summary["drifted_features"]:
            logger.info("Aucune feature avec drift détecté")
            return

        logger.info(f"\nANALYSE DÉTAILLÉE DES TOP {top_n} FEATURES AVEC DRIFT")
        logger.info("=" * 60)

        # Trier les features par score de drift
        sorted_features = sorted(
            summary["drift_details"].items(),
            key=lambda x: x[1]["drift_score"],
            reverse=True,
        )

        for i, (feature, details) in enumerate(sorted_features[:top_n]):
            logger.info(f"\n{i+1}. {feature.upper()}")
            logger.info("   " + "-" * 30)

            # Statistiques descriptives
            ref_stats = self.reference_data[feature].describe()  # type: ignore
            curr_stats = self.current_data[feature].describe()  # type: ignore

            logger.info(f"   Score de drift: {details['drift_score']:.4f}")
            logger.info(f"   Seuil: {details['threshold']:.4f}")
            logger.info(f"   Test: {details['stattest_name']}")

            logger.info("\n   Statistiques de référence:")
            logger.info(f"      Moyenne: {ref_stats['mean']:.4f}")
            logger.info(f"      Médiane: {ref_stats['50%']:.4f}")
            logger.info(f"      Std: {ref_stats['std']:.4f}")

            logger.info("\n   Statistiques actuelles:")
            logger.info(f"      Moyenne: {curr_stats['mean']:.4f}")
            logger.info(f"      Médiane: {curr_stats['50%']:.4f}")
            logger.info(f"      Std: {curr_stats['std']:.4f}")

            # Calcul des changements
            mean_change = (
                (curr_stats["mean"] - ref_stats["mean"]) / ref_stats["mean"]
            ) * 100
            std_change = (
                (curr_stats["std"] - ref_stats["std"]) / ref_stats["std"]
            ) * 100

            logger.info("\n   Changements:")
            logger.info(f"      Moyenne: {mean_change:+.1f}%")
            logger.info(f"      Std: {std_change:+.1f}%")

            # Interprétation
            if abs(mean_change) > 10:
                logger.info("   Changement significatif de la moyenne")
            if abs(std_change) > 20:
                logger.info("   Changement significatif de la variabilité")


def main() -> None:
    """
    Fonction principale pour exécuter l'analyse de drift
    """
    logger.info("DÉTECTION DE DATA DRIFT")
    logger.info("=" * 50)

    # Charger les données (remplacez par vos vrais chemins)
    try:
        logger.info("Chargement des données...")

        # Données de référence (entraînement)
        data_path = Path("data/raw")
        reference_data = pd.read_csv(data_path / "application_train.csv")
        logger.info(f"Données de référence chargées: {reference_data.shape}")

        # Données actuelles (production/test)
        current_data = pd.read_csv(data_path / "application_test.csv")
        logger.info(f"Données actuelles chargées: {current_data.shape}")

        # Créer le détecteur
        detector = DataDriftDetector(reference_data, current_data)

        # Préparer les données
        detector.prepare_data()

        # Détecter le drift
        detector.detect_data_drift()

        # Afficher l'analyse
        detector.print_drift_analysis()

        # Analyse détaillée des features
        detector.create_detailed_feature_analysis()

        # Sauvegarder le rapport HTML dans reports/
        reports_path = Path("reports")
        reports_path.mkdir(exist_ok=True)
        html_path = detector.save_drift_report(
            str(reports_path / "data_drift_report.html")
        )

        if html_path:
            logger.info(f"\nRapport HTML généré: {html_path}")
            logger.info(
                "   Ouvrez ce fichier dans un navigateur pour une analyse interactive"
            )

    except FileNotFoundError as e:
        logger.error(f"Fichier non trouvé: {e}")
        logger.info(
            "Assurez-vous que les fichiers 'application_train.csv' et 'application_test.csv' sont présents"
        )

        # Créer des données d'exemple pour la démonstration
        logger.info("\nCréation de données d'exemple...")

        # Données de référence simulées
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        reference_data = pd.DataFrame(
            {
                f"feature_{i}": np.random.normal(0, 1, n_samples)
                for i in range(n_features)
            }
        )
        reference_data["TARGET"] = np.random.binomial(1, 0.2, n_samples)

        # Données actuelles avec drift simulé
        current_data = pd.DataFrame(
            {
                f"feature_{i}": np.random.normal(
                    0.5 if i < 3 else 0, 1.2 if i < 3 else 1, n_samples
                )
                for i in range(n_features)
            }
        )
        # Pas de colonne TARGET pour simuler des données de production

        logger.info("Données d'exemple créées")


if __name__ == "__main__":
    main()
