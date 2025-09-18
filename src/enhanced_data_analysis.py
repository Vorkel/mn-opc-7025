# enhanced_data_analysis.py
import warnings
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


class HomeCreditDataAnalyzer:
    """
    Analyseur spécialisé pour les données Home Credit Default Risk
    """

    def __init__(self, data_path: str = "data/raw/") -> None:
        """
        Initialise l'analyseur avec le chemin vers les données

        Args:
            data_path (str): Chemin vers le dossier contenant les fichiers CSV
        """
        self.data_path = data_path
        self.main_train: Optional[pd.DataFrame] = None
        self.main_test: Optional[pd.DataFrame] = None
        self.auxiliary_data: Dict[str, pd.DataFrame] = {}

    def load_main_data(self) -> None:
        """
        Charge les fichiers principaux (application_train et application_test)
        """
        print("Chargement des données principales...")

        try:
            # Chargement des données d'entraînement
            self.main_train = pd.read_csv(f"{self.data_path}application_train.csv")
            print(f"application_train.csv chargé: {self.main_train.shape}")

            # Chargement des données de test
            self.main_test = pd.read_csv(f"{self.data_path}application_test.csv")
            print(f"application_test.csv chargé: {self.main_test.shape}")

            # Vérification de la cohérence des colonnes
            train_cols = set(self.main_train.columns)
            test_cols = set(self.main_test.columns)

            common_cols = train_cols.intersection(test_cols)
            train_only = train_cols - test_cols
            test_only = test_cols - train_cols

            print(f"Colonnes communes: {len(common_cols)}")
            print(f"Colonnes uniquement dans train: {train_only}")
            print(f"Colonnes uniquement dans test: {test_only}")

        except FileNotFoundError as e:
            print(f"Erreur: {e}")
            print("Assurez-vous que les fichiers sont dans le bon répertoire")

    def load_auxiliary_data(self) -> None:
        """
        Charge les données auxiliaires (bureau, credit_card, etc.)
        """
        auxiliary_files = {
            "bureau": "bureau.csv",
            "bureau_balance": "bureau_balance.csv",
            "credit_card_balance": "credit_card_balance.csv",
            "installments_payments": "installments_payments.csv",
            "previous_application": "previous_application.csv",
            "pos_cash_balance": "POS_CASH_balance.csv",
        }

        print("\nChargement des données auxiliaires...")

        for name, filename in auxiliary_files.items():
            try:
                df = pd.read_csv(f"{self.data_path}{filename}")
                self.auxiliary_data[name] = df
                print(f"{filename} chargé: {df.shape}")
            except FileNotFoundError:
                print(f"{filename} non trouvé (optionnel)")

    def analyze_target_distribution(self) -> None:
        """
        Analyse la distribution de la variable cible
        """
        if self.main_train is None:
            print("Chargez d'abord les données principales")
            return

        print("\n" + "=" * 50)
        print("ANALYSE DE LA VARIABLE CIBLE")
        print("=" * 50)

        target_counts = self.main_train["TARGET"].value_counts()
        target_pct = self.main_train["TARGET"].value_counts(normalize=True) * 100

        print(f"Distribution de TARGET:")
        print(f"  Classe 0 (bons clients): {target_counts[0]:,} ({target_pct[0]:.2f}%)")
        print(f"  Classe 1 (défauts): {target_counts[1]:,} ({target_pct[1]:.2f}%)")
        print(f"  Ratio de déséquilibre: {target_counts[0]/target_counts[1]:.1f}:1")

        # Visualisation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Graphique en barres
        target_counts.plot(kind="bar", ax=ax1)
        ax1.set_title("Distribution de la variable cible")
        ax1.set_xlabel("Classe")
        ax1.set_ylabel("Nombre de clients")
        ax1.set_xticklabels(["Bons clients", "Défauts"], rotation=0)

        # Graphique en camembert
        ax2.pie(
            target_counts.values,
            labels=["Bons clients", "Défauts"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax2.set_title("Répartition des classes")

        plt.tight_layout()
        plt.show()

    def analyze_missing_values(self) -> None:
        """
        Analyse détaillée des valeurs manquantes
        """
        if self.main_train is None:
            return

        print("\n" + "=" * 50)
        print("ANALYSE DES VALEURS MANQUANTES")
        print("=" * 50)

        # Calcul des valeurs manquantes
        missing_train = self.main_train.isnull().sum()
        missing_pct_train = (missing_train / len(self.main_train)) * 100

        missing_df = pd.DataFrame({
            "Colonne": missing_train.index,
            "Manquantes": missing_train.values,
            "Pourcentage": missing_pct_train.values,
        })

        # Filtrer et trier
        filtered_df = missing_df[missing_df["Manquantes"] > 0]
        if len(filtered_df) > 0:
            # S'assurer que c'est un DataFrame
            if isinstance(filtered_df, pd.Series):
                filtered_df = filtered_df.to_frame().T
            missing_df = filtered_df.sort_values("Pourcentage", ascending=False)
        else:
            missing_df = pd.DataFrame(columns=missing_df.columns)

        print(f"Colonnes avec valeurs manquantes: {len(missing_df)}")
        print("\nTop 15 colonnes avec le plus de valeurs manquantes:")
        print(missing_df.head(15).to_string(index=False, float_format="%.2f"))

        # Visualisation
        if len(missing_df) > 0:
            plt.figure(figsize=(12, 8))
            top_missing = missing_df.head(20)
            sns.barplot(data=top_missing, y="Colonne", x="Pourcentage")
            plt.title("Top 20 - Valeurs manquantes par colonne")
            plt.xlabel("Pourcentage de valeurs manquantes")
            plt.tight_layout()
            plt.show()

    def analyze_numerical_features(self) -> None:
        """
        Analyse des features numériques importantes
        """
        if self.main_train is None:
            return

        print("\n" + "=" * 50)
        print("ANALYSE DES FEATURES NUMÉRIQUES")
        print("=" * 50)

        # Sélectionner les features numériques importantes
        important_numeric = [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "AMT_GOODS_PRICE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "CNT_CHILDREN",
            "CNT_FAM_MEMBERS",
        ]

        existing_numeric = [
            col for col in important_numeric if col in self.main_train.columns
        ]

        if len(existing_numeric) > 0:
            # Statistiques descriptives
            print("Statistiques descriptives des features principales:")
            print(self.main_train[existing_numeric].describe())

            # Visualisation
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()

            for i, col in enumerate(existing_numeric[:4]):
                self.main_train[col].hist(bins=50, ax=axes[i])
                axes[i].set_title(f"Distribution de {col}")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Fréquence")

            plt.tight_layout()
            plt.show()

    def analyze_categorical_features(self) -> None:
        """
        Analyse des features catégorielles
        """
        if self.main_train is None:
            return

        print("\n" + "=" * 50)
        print("ANALYSE DES FEATURES CATÉGORIELLES")
        print("=" * 50)

        # Sélectionner les features catégorielles importantes
        important_categorical = [
            "CODE_GENDER",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS",
            "NAME_HOUSING_TYPE",
        ]

        existing_categorical = [
            col for col in important_categorical if col in self.main_train.columns
        ]

        for col in existing_categorical:
            print(f"\n--- {col} ---")
            value_counts = self.main_train[col].value_counts()
            print(value_counts)

            # Analyse par rapport à la target
            if len(value_counts) <= 10:  # Afficher seulement si pas trop de catégories
                target_analysis = self.main_train.groupby(col)["TARGET"].agg(
                    ["count", "mean"]
                )
                # S'assurer que c'est un DataFrame
                if isinstance(target_analysis, pd.Series):
                    target_analysis = target_analysis.to_frame().T
                elif not isinstance(target_analysis, pd.DataFrame):
                    target_analysis = pd.DataFrame(target_analysis)
                target_analysis.columns = ["Count", "Default_Rate"]
                target_analysis["Default_Rate"] = target_analysis["Default_Rate"] * 100
                print(f"\nTaux de défaut par catégorie:")
                print(target_analysis.sort_values("Default_Rate", ascending=False))

    def create_feature_engineering_suggestions(self) -> None:
        """
        Suggère des transformations de features
        """
        print("\n" + "=" * 50)
        print("SUGGESTIONS DE FEATURE ENGINEERING")
        print("=" * 50)

        suggestions = [
            "VARIABLES NUMÉRIQUES:",
            "   - Transformer DAYS_BIRTH en âge (années)",
            "   - Transformer DAYS_EMPLOYED en années d'expérience",
            "   - Créer ratio AMT_CREDIT / AMT_INCOME_TOTAL",
            "   - Créer ratio AMT_ANNUITY / AMT_INCOME_TOTAL",
            "   - Binning des variables continues",
            "",
            "VARIABLES CATÉGORIELLES:",
            "   - Label encoding pour variables ordinales",
            "   - One-hot encoding pour variables nominales",
            "   - Regrouper catégories rares",
            "",
            "VARIABLES DÉRIVÉES:",
            "   - Indicateurs de valeurs manquantes",
            "   - Interactions entre variables importantes",
            "   - Agrégations des données auxiliaires",
            "",
            "GESTION DU DÉSÉQUILIBRE:",
            "   - SMOTE pour sur-échantillonnage",
            "   - Sous-échantillonnage aléatoire",
            "   - Weights dans les algorithmes",
        ]

        for suggestion in suggestions:
            print(suggestion)

    def generate_data_quality_report(self) -> None:
        """
        Génère un rapport de qualité des données
        """
        if self.main_train is None:
            return

        print("\n" + "=" * 60)
        print("RAPPORT DE QUALITÉ DES DONNÉES")
        print("=" * 60)

        # Métriques générales
        print(f"Nombre d'observations: {len(self.main_train):,}")
        print(f"Nombre de features: {len(self.main_train.columns):,}")
        print(
            "Taille mémoire:"
            f" {self.main_train.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        )

        # Types de données
        print(f"\nTypes de données:")
        dtype_counts = self.main_train.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} colonnes")

        # Valeurs manquantes
        total_missing = self.main_train.isnull().sum().sum()
        missing_pct = (
            total_missing / (len(self.main_train) * len(self.main_train.columns))
        ) * 100
        print(f"\nValeurs manquantes: {total_missing:,} ({missing_pct:.2f}%)")

        # Doublons
        duplicates = self.main_train.duplicated().sum()
        print(f"Doublons: {duplicates:,}")

        # Recommandations
        print(f"\nRECOMMANDATIONS:")
        if missing_pct > 20:
            print(
                "   Taux élevé de valeurs manquantes - Stratégie d'imputation"
                " nécessaire"
            )
        if duplicates > 0:
            print("   Doublons détectés - Nettoyage recommandé")
        if len(self.main_train.columns) > 200:
            print("   Nombreuses features - Sélection de features recommandée")

        print("   Dataset prêt pour la modélisation avec preprocessing approprié")


def main() -> None:
    """
    Fonction principale pour l'analyse complète
    """
    # Créer l'analyseur
    analyzer = HomeCreditDataAnalyzer("data/raw/")

    # Charger les données
    analyzer.load_main_data()
    analyzer.load_auxiliary_data()

    if analyzer.main_train is not None:
        # Analyses principales
        analyzer.analyze_target_distribution()
        analyzer.analyze_missing_values()
        analyzer.analyze_numerical_features()
        analyzer.analyze_categorical_features()
        analyzer.create_feature_engineering_suggestions()
        analyzer.generate_data_quality_report()

        print("\nAnalyse exploratoire terminée!")
        print("Prochaines étapes:")
        print("1. Préprocessing des données")
        print("2. Feature engineering")
        print("3. Entraînement des modèles")
    else:
        print("Impossible de charger les données principales")


if __name__ == "__main__":
    main()
