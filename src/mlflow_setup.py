# mlflow_setup.py
import mlflow
import os
from typing import Optional, Union
from pathlib import Path
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_mlflow(
    tracking_uri: str = "file:mlruns",
    experiment_name: str = "credit_scoring_experiment",
) -> Optional[str]:
    """
    Configuration initiale de MLFlow pour le tracking des expérimentations

    Args:
        tracking_uri (str): URI du serveur MLFlow
        experiment_name (str): Nom de l'expérience MLFlow

    Returns:
        Optional[str]: Nom de l'expérience configurée ou None en cas d'erreur
    """
    try:
        # Configuration du serveur de tracking MLFlow (par défaut local file:mlruns)
        if tracking_uri.startswith("file:"):
            mlruns_path = tracking_uri.split("file:", 1)[1] or "mlruns"
            Path(mlruns_path).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLFlow tracking URI configuré: {tracking_uri}")

        # Vérifier si l'expérience existe déjà
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Créer l'expérience si elle n'existe pas
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Expérience créée avec l'ID: {experiment_id}")
        else:
            logger.info(f"Expérience existante trouvée: {experiment.experiment_id}")

        # Définir l'expérience active
        mlflow.set_experiment(experiment_name)
        logger.info(f"Expérience active définie: {experiment_name}")

        return experiment_name

    except Exception as e:
        logger.error(f"Erreur lors de la configuration MLFlow: {e}")
        return None


def start_mlflow_server(host: str = "0.0.0.0", port: int = 5000) -> None:
    """
    Démarre le serveur MLFlow UI
    Exécutez cette fonction dans un terminal séparé

    Args:
        host (str): Adresse d'écoute du serveur
        port (int): Port d'écoute du serveur
    """
    print("Pour démarrer le serveur MLFlow UI, exécutez dans un terminal :")
    print(f"mlflow ui --host {host} --port {port}")
    print(f"Puis ouvrez http://{host}:{port} dans votre navigateur")


def check_mlflow_connection(tracking_uri: str = "file:mlruns") -> bool:
    """
    Vérifie la connexion au serveur MLFlow

    Args:
        tracking_uri (str): URI du serveur MLFlow à tester

    Returns:
        bool: True si la connexion réussit, False sinon
    """
    try:
        if tracking_uri.startswith("file:"):
            mlruns_path = tracking_uri.split("file:", 1)[1] or "mlruns"
            Path(mlruns_path).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
        # Tenter de récupérer la liste des expériences
        experiments = mlflow.search_experiments()
        logger.info(
            f"Connexion MLFlow réussie, {len(experiments)} expériences trouvées"
        )
        return True
    except Exception as e:
        logger.warning(f"Connexion MLFlow échouée: {e}")
        return False


if __name__ == "__main__":
    # Test de la configuration MLFlow
    experiment_name = setup_mlflow()
    if experiment_name:
        print(f"✅ MLFlow configuré avec succès: {experiment_name}")
        start_mlflow_server()
    else:
        print("❌ Échec de la configuration MLFlow")
        print("Le mode local sera utilisé pour les expérimentations")
