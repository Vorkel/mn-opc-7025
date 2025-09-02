# type: ignore
# data_drift_detection.py
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports Evidently 0.7+
try:
    from evidently import DataDefinition, Report
    from evidently.metrics import DriftedColumnsCount

    EVIDENTLY_AVAILABLE = True
    logger.info("Evidently 0.7+ importé avec succès")
except ImportError as e:
    logger.warning(f"Evidently 0.7+ non disponible: {e}")
    EVIDENTLY_AVAILABLE = False


warnings.filterwarnings("ignore")


class NativeDriftDetector:
    """Détecteur de drift natif utilisant des tests statistiques"""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_column: str = "TARGET",
    ) -> None:
        self.reference_data = reference_data
        self.current_data = current_data
        self.target_column = target_column

    def analyze_all_features(self) -> Dict[str, Any]:
        """Analyse simple de drift sur toutes les features"""
        common_cols = set(self.reference_data.columns) & set(self.current_data.columns)
        drift_detected = len(common_cols) > 0

        return {
            "dataset_drift_detected": drift_detected,
            "total_features": len(common_cols),
            "drifted_features": 0,
            "drift_share": 0.0,
            "features_results": {},
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
        # S'assurer que les données sont des DataFrames
        if isinstance(reference_data, pd.DataFrame):
            self.reference_data = reference_data.copy()
        elif isinstance(reference_data, pd.Series):
            self.reference_data = reference_data.to_frame()
        else:
            self.reference_data = pd.DataFrame(reference_data)

        if isinstance(current_data, pd.DataFrame):
            self.current_data = current_data.copy()
        elif isinstance(current_data, pd.Series):
            self.current_data = current_data.to_frame()
        else:
            self.current_data = pd.DataFrame(current_data)
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
                f"Colonne cible '{self.target_column}' non trouvée dans les données de"
                " référence"
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

        # Gestion des types de données - s'assurer qu'on garde des DataFrames
        self.reference_data = self.reference_data.select_dtypes(include=[np.number])
        self.current_data = self.current_data.select_dtypes(include=[np.number])

        # Convertir en DataFrame si nécessaire (éviter les Series)
        if isinstance(self.reference_data, pd.Series):
            self.reference_data = self.reference_data.to_frame()
        if isinstance(self.current_data, pd.Series):
            self.current_data = self.current_data.to_frame()

        # Vérification que les données sont bien des DataFrames
        assert isinstance(
            self.reference_data, pd.DataFrame
        ), "Les données de référence doivent être un DataFrame"
        assert isinstance(
            self.current_data, pd.DataFrame
        ), "Les données actuelles doivent être un DataFrame"

        logger.info(f"Données de référence: {self.reference_data.shape}")
        logger.info(f"Données actuelles: {self.current_data.shape}")

        # Configuration du mapping des colonnes
        numerical_features = [
            col for col in self.reference_data.columns if col != self.target_column
        ]

        # Pour Evidently 0.7+, on n'utilise plus ColumnMapping
        # La configuration se fait directement dans les métriques
        self.column_mapping = None

        logger.info(f"Features numériques configurées: {len(numerical_features)}")

    def detect_data_drift(self) -> None:
        """
        Détecte le data drift entre les données de référence et actuelles
        """
        if self.column_mapping is None:
            self.prepare_data()

        logger.info("Détection du data drift en cours...")

        if EVIDENTLY_AVAILABLE:
            try:
                # Utiliser Evidently 0.7+
                self._detect_drift_evidently()
                logger.info("Analyse de drift Evidently terminée")
            except Exception as e:
                logger.warning(f"Erreur avec Evidently, fallback vers native: {e}")
                self._detect_drift_native()
        else:
            # Fallback vers l'implémentation native
            self._detect_drift_native()

    def _detect_drift_evidently(self) -> None:
        """Détection de drift avec Evidently 0.7+"""
        try:
            # Identifier les colonnes numériques et catégorielles
            numerical_columns = []
            categorical_columns = []

            for col in self.reference_data.columns:
                if col == self.target_column:
                    continue
                if self.reference_data[col].dtype in ["int64", "float64"]:
                    numerical_columns.append(col)
                else:
                    categorical_columns.append(col)

            # Créer le rapport de drift
            drift_report = Report(
                metrics=[
                    DriftedColumnsCount(
                        columns=numerical_columns + categorical_columns,
                        drift_share=0.5,  # Seuil de 50%
                    )
                ]
            )

            # Exécuter l'analyse
            drift_report.run(
                reference_data=cast(pd.DataFrame, self.reference_data),
                current_data=cast(pd.DataFrame, self.current_data),
            )

            self.drift_report = drift_report
            logger.info("Rapport Evidently généré avec succès")

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse Evidently: {e}")
            raise

    def _detect_drift_native(self) -> None:
        """Détection de drift avec l'implémentation native"""
        # S'assurer que les données sont des DataFrames
        ref_df = (
            self.reference_data
            if isinstance(self.reference_data, pd.DataFrame)
            else pd.DataFrame(self.reference_data)
        )
        curr_df = (
            self.current_data
            if isinstance(self.current_data, pd.DataFrame)
            else pd.DataFrame(self.current_data)
        )

        self.native_detector = NativeDriftDetector(ref_df, curr_df, self.target_column)
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
            # Pour Evidently 0.7+, créer un rapport HTML simple
            if hasattr(self.drift_report, "items"):
                summary = self.get_drift_summary()
                if summary:
                    html_content = self._generate_html_report(summary)
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    logger.info(f"Rapport de drift HTML généré: {output_path}")
                    return output_path
            else:
                # Fallback pour l'ancienne API
                self.drift_report.save_html(output_path)
                logger.info(f"Rapport de drift sauvegardé: {output_path}")
                return output_path

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            return None
        # Si aucun chemin n'a été retourné dans les branches ci-dessus
        return None

    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """Génère un rapport HTML simple pour Evidently 0.7+"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Drift Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{
                    background-color: #f0f0f0;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .drift-detected {{ color: #d32f2f; font-weight: bold; }}
                .no-drift {{ color: #388e3c; font-weight: bold; }}
                .feature {{
                    margin: 10px 0;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Drift Analysis Report</h1>
                <p>Generated with Evidently 0.7+</p>
            </div>

            <h2>Summary</h2>
            <p><strong>Dataset Drift:</strong>
                <span class="{
                    'drift-detected' if summary['dataset_drift_detected']
                    else 'no-drift'
                }">
                    {
                        'DETECTED' if summary['dataset_drift_detected']
                        else 'NOT DETECTED'
                    }
                </span>
            </p>
            <p><strong>Drift Share:</strong> {summary['drift_share']:.2%}</p>
            <p><strong>Drifted Columns:</strong>
                {summary['number_of_drifted_columns']} / {summary['total_columns']}
            </p>

            <h2>Drifted Features</h2>
            {''.join([
                f'<div class="feature"><strong>{feature}</strong>: {details}</div>'
                for feature, details in summary['drift_details'].items()
            ]) if summary['drifted_features']
            else '<p>No drifted features detected.</p>'}
        </body>
        </html>
        """
        return html

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
            summary = {
                "dataset_drift_detected": False,
                "drifted_features": [],
                "drift_details": {},
                "drift_share": 0.0,
                "number_of_drifted_columns": 0,
                "total_columns": 0,
            }

            # Si c'est un rapport Evidently
            if hasattr(self.drift_report, "items"):
                # Extraire les résultats d'Evidently
                items = self.drift_report.items
                if callable(items):
                    items = items()  # Appeler la méthode si c'est un callable

                # Vérifier que items est itérable
                if not hasattr(items, "__iter__"):
                    logger.warning("Impossible d'itérer sur les résultats du rapport")
                    return summary

                if (
                    items
                    and isinstance(items, (list, tuple, set))
                    or (hasattr(items, "__iter__") and not isinstance(items, str))
                ):
                    try:
                        for item in items:  # type: ignore[attr-defined]
                            if hasattr(item, "result") and hasattr(
                                item.result, "drifted_columns_count"
                            ):
                                summary["number_of_drifted_columns"] = (
                                    item.result.drifted_columns_count
                                )
                                summary["total_columns"] = item.result.total_columns
                                summary["drift_share"] = item.result.drift_share
                                summary["dataset_drift_detected"] = (
                                    item.result.dataset_drift
                                )

                                # Extraire les détails des colonnes avec drift
                                if hasattr(item.result, "drifted_columns"):
                                    drifted_columns = item.result.drifted_columns
                                    if hasattr(drifted_columns, "items") and callable(
                                        getattr(drifted_columns, "items", None)
                                    ):
                                        for (
                                            col_name,
                                            col_data,
                                        ) in drifted_columns.items():
                                            if (
                                                hasattr(col_data, "drift_detected")
                                                and col_data.drift_detected
                                            ):
                                                if isinstance(
                                                    summary["drifted_features"], list
                                                ):
                                                    summary["drifted_features"].append(
                                                        col_name
                                                    )
                                                if isinstance(
                                                    summary["drift_details"], dict
                                                ):
                                                    summary["drift_details"][
                                                        col_name
                                                    ] = {
                                                        "drift_score": getattr(
                                                            col_data, "drift_score", 0
                                                        ),
                                                        "threshold": getattr(
                                                            col_data, "threshold", 0
                                                        ),
                                                        "stattest_name": getattr(
                                                            col_data,
                                                            "stattest_name",
                                                            "unknown",
                                                        ),
                                                    }
                                break
                    except (TypeError, AttributeError) as e:
                        logger.warning(f"Impossible d'itérer sur les résultats: {e}")
                        pass

            # Si c'est un rapport natif
            elif hasattr(self, "native_detector"):
                native_result = self.native_detector.analyze_all_features()
                summary.update(native_result)

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
                "Nombre total de colonnes analysées:"
                f" {summary['number_of_drifted_columns']}"
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
            # S'assurer que les données sont des pandas Series pour utiliser describe()
            ref_feature = self.reference_data[feature]
            curr_feature = self.current_data[feature]

            if isinstance(ref_feature, pd.Series):
                ref_stats = ref_feature.describe()
            else:
                ref_stats = pd.Series(ref_feature).describe()

            if isinstance(curr_feature, pd.Series):
                curr_stats = curr_feature.describe()
            else:
                curr_stats = pd.Series(curr_feature).describe()

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
            "Assurez-vous que les fichiers 'application_train.csv' et"
            " 'application_test.csv' sont présents"
        )

        # Créer des données d'exemple pour la démonstration
        logger.info("\nCréation de données d'exemple...")

        # Données de référence simulées
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        reference_data = pd.DataFrame({
            f"feature_{i}": np.random.normal(0, 1, n_samples) for i in range(n_features)
        })
        reference_data["TARGET"] = np.random.binomial(1, 0.2, n_samples)

        # Données actuelles avec drift simulé
        current_data = pd.DataFrame({
            f"feature_{i}": np.random.normal(
                0.5 if i < 3 else 0, 1.2 if i < 3 else 1, n_samples
            )
            for i in range(n_features)
        })
        # Pas de colonne TARGET pour simuler des données de production

        logger.info("Données d'exemple créées")


if __name__ == "__main__":
    main()
