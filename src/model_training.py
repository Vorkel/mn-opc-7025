# model_training.py
import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    make_scorer,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    accuracy_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
import logging
from contextlib import nullcontext

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Import des classes personnalisées avec fallback
try:
    from src.business_score import BusinessScorer
except ImportError:
    try:
        from src.business_score import BusinessScorer
    except ImportError:
        logger.warning(
            "BusinessScorer non disponible, certaines fonctionnalités seront limitées"
        )
        BusinessScorer = None


class ModelTrainer:
    """
    Classe pour l'entraînement et l'évaluation des modèles avec MLFlow
    """

    def __init__(self, experiment_name: str = "credit_scoring_experiment") -> None:
        """
        Initialise le trainer

        Args:
            experiment_name (str): Nom de l'expérience MLFlow
        """
        self.experiment_name = experiment_name
        self.scorer = BusinessScorer(cost_fn=10, cost_fp=1) if BusinessScorer else None
        self.models: Dict[str, Dict[str, Any]] = {}
        self.best_model: Optional[Any] = None
        self.best_threshold: Optional[float] = None

        # Configuration MLflow: URI depuis env, défaut local file:mlruns
        uri = os.getenv("MLFLOW_TRACKING_URI", "file:mlruns")
        try:
            # Si file:mlruns → s'assurer que le dossier existe
            if uri.startswith("file:"):
                mlruns_path = uri.split("file:", 1)[1] or "mlruns"
                Path(mlruns_path).mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment_name)
            self.mlflow_enabled = True
            logger.info(f"MLFlow configuré — URI: {uri}, exp: {experiment_name}")
        except Exception as e:
            logger.warning(f"MLFlow non accessible ({str(e)}), mode local désactivé")
            self.mlflow_enabled = False

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prépare les données pour l'entraînement

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Variable cible
            test_size (float): Taille du set de test
            random_state (int): Seed pour la reproductibilité

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
        """
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Taille des données d'entraînement: {X_train.shape}")  # type: ignore
        logger.info(f"Taille des données de test: {X_test.shape}")  # type: ignore
        # Distribution des classes avec numpy
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        logger.info(
            f"Distribution des classes (train): {dict(zip(unique_train, counts_train))}"
        )
        logger.info(
            f"Distribution des classes (test): {dict(zip(unique_test, counts_test))}"
        )

        return X_train, X_test, y_train, y_test  # type: ignore

    def create_business_scorer(self) -> Any:
        """
        Crée un scorer personnalisé pour GridSearchCV

        Returns:
            Any: Scorer personnalisé
        """
        if not self.scorer:
            logger.error("BusinessScorer non disponible")
            return None

        def business_score_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            # Inverser le signe car GridSearchCV maximise le score
            if self.scorer:
                return -self.scorer.calculate_business_cost(y_true, y_pred)
            return 0.0

        return make_scorer(business_score_func, greater_is_better=True)

    def train_baseline_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Entraîne un modèle de base (régression logistique simple)

        Args:
            X_train (pd.DataFrame): Données d'entraînement
            y_train (pd.Series): Labels d'entraînement
            X_test (pd.DataFrame): Données de test
            y_test (pd.Series): Labels de test

        Returns:
            Tuple[Any, Dict[str, Any]]: Modèle entraîné et métriques
        """
        # Modèle de base
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Prédictions
        if model is not None and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError("Le modèle ne supporte pas predict_proba")

        # Évaluation
        if self.scorer:
            metrics = self.scorer.evaluate_model(np.array(y_test), y_proba)  # type: ignore
        else:
            logger.error("Scorer non disponible pour l'évaluation")
            return model, {}

        # Logging MLFlow conditionnel
        if self.mlflow_enabled:
            try:
                with mlflow.start_run(run_name="baseline_logistic_regression"):
                    mlflow.log_param("model_type", "LogisticRegression")
                    mlflow.log_param("baseline", True)
                    self.log_metrics(metrics)

                    # Sauvegarder le modèle
                    mlflow.sklearn.log_model(model, "model")  # type: ignore
            except Exception as e:
                logger.error(f"MLFlow logging échoué: {str(e)}")

        self.models["baseline"] = {
            "model": model,
            "metrics": metrics,
            "threshold": metrics["threshold"],
        }

        logger.info("Modèle de base entraîné avec succès!")
        if self.scorer:
            self.scorer.print_evaluation_report(metrics)

        return model, metrics

    def train_with_sampling(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sampling_strategy: str = "smote",
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Entraîne des modèles avec différentes stratégies de rééquilibrage

        Args:
            X_train (pd.DataFrame): Données d'entraînement
            y_train (pd.Series): Labels d'entraînement
            X_test (pd.DataFrame): Données de test
            y_test (pd.Series): Labels de test
            sampling_strategy (str): 'smote', 'undersample', ou 'none'

        Returns:
            Tuple[Any, Dict[str, Any]]: Modèle entraîné et métriques
        """
        run_name = f"random_forest_{sampling_strategy}"

        # Logging MLFlow conditionnel
        if self.mlflow_enabled:
            try:
                mlflow_context = mlflow.start_run(run_name=run_name)
            except Exception as e:
                logger.error(f"MLFlow échoué: {str(e)}")
                mlflow_context = None
        else:
            mlflow_context = None

        # Configuration du pipeline selon la stratégie
        if sampling_strategy == "smote":
            sampler = SMOTE(random_state=42)
            if self.mlflow_enabled:
                mlflow.log_param("sampling_strategy", "SMOTE")
        elif sampling_strategy == "undersample":
            sampler = RandomUnderSampler(random_state=42)
            if self.mlflow_enabled:
                mlflow.log_param("sampling_strategy", "RandomUnderSampler")
        else:
            sampler = None
            if self.mlflow_enabled:
                mlflow.log_param("sampling_strategy", "None")

        # Créer le pipeline
        if sampler:
            pipeline = ImbPipeline(
                [
                    ("sampler", sampler),
                    ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
                ]
            )
        else:
            pipeline = RandomForestClassifier(random_state=42, n_jobs=-1)

        # Entraînement
        pipeline.fit(X_train, y_train)

        # Prédictions
        if pipeline is not None and hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]  # type: ignore
        else:
            raise ValueError("Le pipeline ne supporte pas predict_proba")

        # Évaluation
        if self.scorer:
            metrics = self.scorer.evaluate_model(np.array(y_test), y_proba)  # type: ignore
        else:
            logger.error("Scorer non disponible pour l'évaluation")
            return pipeline, {}

        # Logging MLFlow
        if self.mlflow_enabled:
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            self.log_metrics(metrics)

            # Sauvegarder le modèle
            try:
                mlflow.sklearn.log_model(pipeline, "model")  # type: ignore
            except Exception as e:
                logger.error(f"Erreur MLFlow log_model: {e}")

        self.models[f"rf_{sampling_strategy}"] = {
            "model": pipeline,
            "metrics": metrics,
            "threshold": metrics["threshold"],
        }

        logger.info(f"Modèle Random Forest ({sampling_strategy}) entraîné avec succès!")
        if self.scorer:
            self.scorer.print_evaluation_report(metrics)

        return pipeline, metrics

    def train_lgbm_with_gridsearch(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Entraîne un modèle LightGBM avec optimisation des hyperparamètres

        Args:
            X_train (pd.DataFrame): Données d'entraînement
            y_train (pd.Series): Labels d'entraînement
            X_test (pd.DataFrame): Données de test
            y_test (pd.Series): Labels de test

        Returns:
            Tuple[Any, Dict[str, Any]]: Modèle entraîné et métriques
        """
        # Context manager conditionnel pour MLFlow
        mlflow_context = (
            mlflow.start_run(run_name="lightgbm_gridsearch")
            if self.mlflow_enabled
            else nullcontext()
        )

        with mlflow_context:
            # Paramètres à optimiser (mode rapide via env FAST_TRAINING=1)
            fast = os.getenv("FAST_TRAINING", "0") == "1"
            if fast:
                param_grid = {
                    "n_estimators": [100],
                    "learning_rate": [0.1],
                    "max_depth": [3, 5],
                    "num_leaves": [31],
                    "min_child_samples": [20],
                }
            else:
                param_grid = {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "num_leaves": [31, 50],
                    "min_child_samples": [20, 30],
                }

            # Modèle de base
            lgbm = LGBMClassifier(
                random_state=42, objective="binary", metric="binary_logloss", verbose=-1
            )

            # Scorer personnalisé
            business_scorer = self.create_business_scorer()

            # GridSearchCV
            grid_search = GridSearchCV(
                lgbm, param_grid, scoring=business_scorer, cv=3, n_jobs=-1, verbose=1
            )

            logger.info("Démarrage de la recherche d'hyperparamètres...")
            grid_search.fit(X_train, y_train)

            # Meilleur modèle
            best_model = grid_search.best_estimator_

            # Prédictions
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
            else:
                raise ValueError("Le modèle ne supporte pas predict_proba")

            # Évaluation
            if self.scorer:
                metrics = self.scorer.evaluate_model(np.array(y_test), y_proba)  # type: ignore
            else:
                logger.error("Scorer non disponible pour l'évaluation")
                return best_model, {}

            # Logging MLFlow
            if self.mlflow_enabled:
                mlflow.log_param("model_type", "LightGBM")
                mlflow.log_param("gridsearch", True)

                # Logger les meilleurs hyperparamètres
                for param, value in grid_search.best_params_.items():
                    mlflow.log_param(f"best_{param}", value)

                mlflow.log_metric("best_cv_score", grid_search.best_score_)

                self.log_metrics(metrics)

                # Sauvegarder le modèle
                mlflow.sklearn.log_model(best_model, "model")  # type: ignore

            self.models["lgbm_optimized"] = {
                "model": best_model,
                "metrics": metrics,
                "threshold": metrics["threshold"],
                "best_params": grid_search.best_params_,
            }

            logger.info("Modèle LightGBM optimisé entraîné avec succès!")
            logger.info(f"Meilleurs paramètres: {grid_search.best_params_}")
            if self.scorer:
                self.scorer.print_evaluation_report(metrics)

            return best_model, metrics

    def compare_models(self) -> Optional[pd.DataFrame]:
        """
        Compare tous les modèles entraînés

        Returns:
            Optional[pd.DataFrame]: DataFrame de comparaison ou None
        """
        if not self.models:
            logger.warning("Aucun modèle n'a été entraîné!")
            return None

        logger.info("=" * 80)
        logger.info("COMPARAISON DES MODÈLES")
        logger.info("=" * 80)

        # Créer un DataFrame de comparaison
        comparison_data = []

        for model_name, model_info in self.models.items():
            metrics = model_info["metrics"]
            comparison_data.append(
                {
                    "Modèle": model_name,
                    "Coût Métier": metrics["business_cost"],
                    "AUC": metrics["auc_score"],
                    "Accuracy": metrics["accuracy"],
                    "Précision": metrics["precision"],
                    "Rappel": metrics["recall"],
                    "F1-Score": metrics["f1_score"],
                    "Seuil Optimal": metrics["threshold"],
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("Coût Métier")

        logger.info(comparison_df.to_string(index=False, float_format="%.4f"))

        # Identifier le meilleur modèle
        best_model_name = comparison_df.iloc[0]["Modèle"]
        self.best_model = self.models[best_model_name]["model"]
        self.best_threshold = self.models[best_model_name]["threshold"]

        logger.info(f"🏆 MEILLEUR MODÈLE: {best_model_name}")
        logger.info(
            f"Coût métier le plus bas: {comparison_df.iloc[0]['Coût Métier']:.2f}"
        )

        return comparison_df

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log les métriques dans MLFlow (conditionnel)

        Args:
            metrics (Dict[str, Any]): Métriques à logger
        """
        if not self.mlflow_enabled:
            return

        try:
            mlflow.log_metric("business_cost", metrics["business_cost"])
            mlflow.log_metric("auc_score", metrics["auc_score"])
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("f1_score", metrics["f1_score"])
            mlflow.log_metric("optimal_threshold", metrics["threshold"])
        except Exception as e:
            logger.error(f"MLFlow metrics logging échoué: {str(e)}")

        # Matrice de confusion
        cm = metrics["confusion_matrix"]
        mlflow.log_metric("true_negatives", cm["tn"])
        mlflow.log_metric("false_positives", cm["fp"])
        mlflow.log_metric("false_negatives", cm["fn"])
        mlflow.log_metric("true_positives", cm["tp"])

    def save_best_model(
        self, filepath: str = "models/best_credit_model.pkl"
    ) -> Optional[str]:
        """
        Sauvegarde le meilleur modèle avec pathlib

        Args:
            filepath (str): Chemin de sauvegarde

        Returns:
            Optional[str]: Chemin de sauvegarde ou None
        """
        if self.best_model is None:
            logger.error("Aucun modèle n'a été sélectionné comme meilleur!")
            return None

        # Créer le dossier models/ si nécessaire
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.best_model,
            "threshold": self.best_threshold,
            "feature_names": getattr(self, "feature_names", []),
            "scaler": getattr(self, "scaler", None),
        }

        joblib.dump(model_data, output_path)
        logger.info(f"Meilleur modèle sauvegardé dans: {output_path}")
        return str(output_path)

    def feature_importance_analysis(
        self, X_train: pd.DataFrame, feature_names: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Analyse l'importance des features du meilleur modèle

        Args:
            X_train (pd.DataFrame): Données d'entraînement
            feature_names (Optional[List[str]]): Noms des features

        Returns:
            Optional[pd.DataFrame]: DataFrame d'importance ou None
        """
        if self.best_model is None:
            logger.error("Aucun modèle n'a été sélectionné!")
            return None

        # Vérifier si le modèle a un attribut feature_importances_
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, "coef_"):
            importances = np.abs(self.best_model.coef_[0])
        elif (
            hasattr(self.best_model, "named_steps")
            and "classifier" in self.best_model.named_steps
        ):
            # Pour les pipelines
            classifier = self.best_model.named_steps["classifier"]
            if hasattr(classifier, "feature_importances_"):
                importances = classifier.feature_importances_
            elif hasattr(classifier, "coef_"):
                importances = np.abs(classifier.coef_[0])
            else:
                logger.warning(
                    "Le modèle ne supporte pas l'analyse d'importance des features"
                )
                return None
        else:
            logger.warning(
                "Le modèle ne supporte pas l'analyse d'importance des features"
            )
            return None

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        # Créer un DataFrame d'importance
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        logger.info("=" * 50)
        logger.info("TOP 20 FEATURES LES PLUS IMPORTANTES")
        logger.info("=" * 50)
        logger.info(importance_df.head(20).to_string(index=False, float_format="%.6f"))

        return importance_df


# Exemple d'utilisation complète
if __name__ == "__main__":
    # Configuration MLflow: respecter MLFLOW_TRACKING_URI (défaut local file:mlruns)
    uri = os.getenv("MLFLOW_TRACKING_URI", "file:mlruns")
    try:
        if uri.startswith("file:"):
            mlruns_path = uri.split("file:", 1)[1] or "mlruns"
            Path(mlruns_path).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)
        logger.info(f"MLFlow URI configuré: {uri}")
    except Exception as e:
        logger.warning(f"Configuration MLFlow ignorée: {str(e)}")

    # Créer le trainer
    trainer = ModelTrainer("credit_scoring_experiment")

    # Simuler des données (remplacez par vos vraies données)
    np.random.seed(42)
    n_samples = 10000
    n_features = 20

    # Créer des données simulées
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],  # type: ignore
    )
    y = pd.Series(np.random.binomial(1, 0.2, n_samples))  # 20% de défauts

    # Préparer les données
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)

    # Entraîner différents modèles
    logger.info("1. Entraînement du modèle de base...")
    trainer.train_baseline_model(X_train, y_train, X_test, y_test)

    logger.info("\n2. Entraînement avec SMOTE...")
    trainer.train_with_sampling(X_train, y_train, X_test, y_test, "smote")

    logger.info("\n3. Entraînement avec sous-échantillonnage...")
    trainer.train_with_sampling(X_train, y_train, X_test, y_test, "undersample")

    logger.info("\n4. Entraînement LightGBM avec optimisation...")
    trainer.train_lgbm_with_gridsearch(X_train, y_train, X_test, y_test)

    # Comparer les modèles
    comparison_df = trainer.compare_models()

    # Analyser l'importance des features
    importance_df = trainer.feature_importance_analysis(X_train, list(X.columns))

    # Sauvegarder le meilleur modèle dans models/
    trainer.save_best_model("models/best_credit_model.pkl")

    logger.info(
        "\n✅ Entraînement terminé! Consultez MLFlow UI sur http://localhost:5000"
    )
