# Notebook: 01_data_exploration.ipynb
# Exploration des données Home Credit Default Risk

"""
OBJECTIFS DE CE NOTEBOOK :
- Explorer les données Home Credit de manière interactive
- Comprendre la structure et la qualité des données
- Analyser la variable cible et les features principales
- Identifier les problèmes potentiels (valeurs manquantes, outliers)
- Préparer les insights pour la modélisation
"""

# =============================================================================
# Configuration et imports
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Créer le dossier reports/ si nécessaire
reports_path = Path("reports")
reports_path.mkdir(exist_ok=True)

# Configuration des graphiques
try:
    plt.style.use("seaborn-v0_8")
except OSError:
    plt.style.use("seaborn")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)

# Configuration Plotly
import plotly.io as pio

pio.templates.default = "plotly_white"

print("Notebook d'exploration des données Home Credit")
print("=" * 50)
print(f"Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# Chargement des données principales
# =============================================================================
print("Chargement des données...")

# Chargement des données principales
try:
    # Chargement des données avec chemins corrects
    df_train = pd.read_csv("data/raw/application_train.csv")
    df_test = pd.read_csv("data/raw/application_test.csv")
    print("Données chargées avec succès!")
except FileNotFoundError:
    print("Erreur lors du chargement des données")
    print(
        "Assurez-vous que les fichiers sont dans data/raw/ et que vous exécutez depuis la racine"
    )

# Vérification des colonnes
train_cols = set(df_train.columns)
test_cols = set(df_test.columns)
common_cols = train_cols.intersection(test_cols)

print(f"Colonnes communes: {len(common_cols)}")
print(f"Colonnes uniquement dans train: {train_cols - test_cols}")
print(f"Colonnes uniquement dans test: {test_cols - train_cols}")

# =============================================================================
# Vue d'ensemble des données
# =============================================================================
print("\nVUE D'ENSEMBLE DES DONNÉES")
print("=" * 40)

# Informations générales
print(f"Nombre d'observations (train): {len(df_train):,}")
print(f"Nombre d'observations (test): {len(df_test):,}")
print(f"Nombre de features: {len(df_train.columns):,}")
print(
    f"Taille mémoire (train): {df_train.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
)

# Types de données
print(f"\nTypes de données:")
dtype_counts = df_train.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"   {dtype}: {count} colonnes")

# Aperçu des premières lignes
print(f"\nAperçu des données:")
print(df_train.info())

# =============================================================================
# Analyse de la variable cible
# =============================================================================
print("\nANALYSE DE LA VARIABLE CIBLE")
print("=" * 40)

# Distribution de la variable cible
target_counts = df_train["TARGET"].value_counts()
target_pct = df_train["TARGET"].value_counts(normalize=True) * 100

print(f"Distribution de TARGET:")
print(f"  Classe 0 (bons clients): {target_counts[0]:,} ({target_pct[0]:.2f}%)")
print(f"  Classe 1 (défauts): {target_counts[1]:,} ({target_pct[1]:.2f}%)")
print(f"  Ratio de déséquilibre: {target_counts[0]/target_counts[1]:.1f}:1")

# Visualisation interactive
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Distribution des classes", "Pourcentage par classe"),
    specs=[[{"type": "bar"}, {"type": "pie"}]],
)

# Graphique en barres
fig.add_trace(
    go.Bar(
        x=["Bons clients", "Défauts"],
        y=target_counts.values,
        name="Nombre",
        marker_color=["lightgreen", "lightcoral"],
    ),
    row=1,
    col=1,
)

# Graphique en camembert
fig.add_trace(
    go.Pie(
        labels=["Bons clients", "Défauts"],
        values=target_counts.values,
        name="Pourcentage",
    ),
    row=1,
    col=2,
)

fig.update_layout(title_text="Analyse de la variable cible", height=500)

# Sauvegarder le graphique
try:
    fig.write_html("reports/target_analysis.html")
    print("Graphique HTML sauvegardé avec succès")
except Exception as e:
    print(f"Erreur lors de la sauvegarde HTML: {e}")

# Tentative de sauvegarde PNG avec gestion d'erreur robuste
try:
    fig.write_image("reports/target_analysis.png")
    print("Graphique PNG sauvegardé avec succès")
except Exception as e:
    print(f"Erreur lors de la sauvegarde PNG: {e}")
    print("Les graphiques HTML sont disponibles dans le dossier reports/")

# =============================================================================
# Analyse des valeurs manquantes
# =============================================================================
print("\nANALYSE DES VALEURS MANQUANTES")
print("=" * 40)

# Calcul des valeurs manquantes
missing_data = df_train.isnull().sum()
missing_pct = (missing_data / len(df_train)) * 100

missing_df = pd.DataFrame(
    {
        "Colonne": missing_data.index,
        "Manquantes": missing_data.values,
        "Pourcentage": missing_pct.values,
    }
)

# Filtrer et trier
missing_df = missing_df[missing_df["Manquantes"] > 0].sort_values( # type: ignore
    "Pourcentage", ascending=False
)

print(
    f"Colonnes avec valeurs manquantes: {len(missing_df)} sur {len(df_train.columns)}"
)
print(f"Pourcentage global de valeurs manquantes: {missing_pct.sum():.2f}%")

# Top 15 colonnes avec le plus de valeurs manquantes
print(f"\nTop 15 colonnes avec valeurs manquantes:")
print(missing_df.head(20))

# Visualisation interactive des valeurs manquantes
if len(missing_df) > 0:
    fig = px.bar(
        missing_df.head(20),
        x="Pourcentage",
        y="Colonne",
        title="Top 20 - Valeurs manquantes par colonne",
        labels={"Pourcentage": "Pourcentage de valeurs manquantes"},
        color="Pourcentage",
        color_continuous_scale="Reds",
    )
    fig.update_layout(height=600)
    # Sauvegarder le graphique
    try:
        fig.write_html("reports/missing_values_analysis.html")
        print("Graphique HTML sauvegardé avec succès")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde HTML: {e}")

    try:
        fig.write_image("reports/missing_values_analysis.png")
        print("Graphique PNG sauvegardé avec succès")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde PNG: {e}")

# =============================================================================
# Analyse des features numériques principales
# =============================================================================
print("\nANALYSE DES FEATURES NUMÉRIQUES")
print("=" * 40)

# Sélection des features numériques importantes
important_numeric = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
]

# Vérifier l'existence des colonnes
existing_numeric = [col for col in important_numeric if col in df_train.columns]
print(f"Features numériques analysées: {len(existing_numeric)}")

# Statistiques descriptives
print("\nStatistiques descriptives:")
stats = df_train[existing_numeric].describe()
print(stats)

# Transformation des variables temporelles
df_analysis = df_train.copy()
df_analysis["AGE_YEARS"] = -df_analysis["DAYS_BIRTH"] / 365.25
df_analysis["EMPLOYMENT_YEARS"] = -df_analysis["DAYS_EMPLOYED"] / 365.25
df_analysis["YEARS_SINCE_REGISTRATION"] = -df_analysis["DAYS_REGISTRATION"] / 365.25

# Visualisation des distributions
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.ravel()

# Features transformées et importantes
plot_features = [
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "CNT_CHILDREN",
]

for i, feature in enumerate(plot_features):
    if feature in df_analysis.columns:
        # Histogramme
        df_analysis[feature].hist(bins=50, ax=axes[i], alpha=0.7)
        axes[i].set_title(f"Distribution de {feature}")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Fréquence")
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/numeric_features_distribution.png", dpi=300, bbox_inches="tight")
print("Graphique matplotlib généré et sauvegardé")

# =============================================================================
# Analyse des features catégorielles
# =============================================================================
print("\nANALYSE DES FEATURES CATÉGORIELLES")
print("=" * 40)

# Features catégorielles importantes
important_categorical = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "NAME_CONTRACT_TYPE",
    "ORGANIZATION_TYPE",
]

existing_categorical = [col for col in important_categorical if col in df_train.columns]

# Analyse de chaque feature catégorielle
for feature in existing_categorical[:6]:  # Limiter à 6 pour l'affichage
    print(f"\n--- {feature} ---")

    # Comptage des valeurs
    value_counts = df_train[feature].value_counts()
    print(f"Nombre de catégories: {len(value_counts)}")
    print(f"Top 5 catégories:")
    print(value_counts.head())

    # Analyse par rapport à la target
    if len(value_counts) <= 15:  # Afficher seulement si pas trop de catégories
        target_analysis = (
            df_train.groupby(feature)["TARGET"].agg(["count", "mean"]).round(4)
        )
        target_analysis.columns = ["Count", "Default_Rate"]
        target_analysis = target_analysis.sort_values("Default_Rate", ascending=False) # type: ignore

        print(f"\nTaux de défaut par catégorie:")
        print(target_analysis)

        # Visualisation
        fig = px.bar(
            x=target_analysis.index,
            y=target_analysis["Default_Rate"],
            title=f"Taux de défaut par {feature}",
            labels={"y": "Taux de défaut", "x": feature},
        )
        fig.update_layout(height=400)
        # Sauvegarder le graphique
        try:
            fig.write_html(f"reports/target_analysis_{feature}.html")
            print(f"Graphique HTML sauvegardé pour {feature}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde HTML pour {feature}: {e}")

        try:
            fig.write_image(f"reports/target_analysis_{feature}.png")
            print(f"Graphique PNG sauvegardé pour {feature}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde PNG pour {feature}: {e}")
        print(f"Graphique généré (taux de défaut pour {feature})")

# =============================================================================
# Analyse des corrélations
# =============================================================================
print("\nANALYSE DES CORRÉLATIONS")
print("=" * 40)

# Sélection des features numériques pour la corrélation
numeric_features = df_train.select_dtypes(include=[np.number]).columns.tolist()

# Enlever SK_ID_CURR qui n'est pas utile pour la corrélation
numeric_features = [col for col in numeric_features if col != "SK_ID_CURR"]

# Limiter à 20 features pour la lisibilité
numeric_features = numeric_features[:20]

# Matrice de corrélation
corr_matrix = df_train[numeric_features].corr() # type: ignore

# Visualisation avec seaborn
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
)
plt.title("Matrice de corrélation des features numériques")
plt.tight_layout()
plt.savefig("reports/correlation_matrix.png", dpi=300, bbox_inches="tight")
print("Graphique matplotlib généré et sauvegardé")

# Corrélations les plus fortes avec TARGET
if "TARGET" in corr_matrix.columns:
    target_corr = (
        corr_matrix["TARGET"].abs().sort_values(ascending=False)[1:11] # type: ignore
    )  # Top 10
    print(f"\nTop 10 corrélations avec TARGET:")
    for feature, corr in target_corr.items(): # type: ignore
        print(f"  {feature}: {corr:.4f}")

# =============================================================================
# Détection des outliers
# =============================================================================
print("\nDÉTECTION DES OUTLIERS")
print("=" * 40)


# Fonction pour détecter les outliers
def detect_outliers(
    df: pd.DataFrame, features: List[str], method: str = "iqr"
) -> Dict[str, Dict[str, Any]]:
    """Détecte les outliers avec la méthode IQR"""
    outliers_dict = {}

    for feature in features:
        if feature in df.columns:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
            outliers_dict[feature] = {
                "count": len(outliers),
                "percentage": (len(outliers) / len(df)) * 100,
                "bounds": (lower_bound, upper_bound),
            }

    return outliers_dict


# Analyse des outliers pour les features importantes
outlier_features = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AGE_YEARS"]
outliers_info = detect_outliers(df_analysis, outlier_features)

print("Outliers détectés:")
for feature, info in outliers_info.items():
    print(f"{feature}: {info['count']} outliers ({info['percentage']:.2f}%)")

# Visualisation des outliers
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for i, feature in enumerate(outlier_features):
    if feature in df_analysis.columns:
        # Boxplot
        df_analysis[feature].plot.box(ax=axes[i])
        axes[i].set_title(f"Boxplot - {feature}")
        axes[i].set_ylabel(feature)
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/outliers_analysis.png", dpi=300, bbox_inches="tight")
print("Graphique matplotlib généré et sauvegardé")

# =============================================================================
# Analyse des données temporelles
# =============================================================================
print("\nANALYSE DES DONNÉES TEMPORELLES")
print("=" * 40)

# Analyse de l'âge des clients
age_stats = df_analysis["AGE_YEARS"].describe()
print(f"Statistiques d'âge:")
print(f"  Moyenne: {age_stats['mean']:.1f} ans")
print(f"  Médiane: {age_stats['50%']:.1f} ans")
print(f"  Min: {age_stats['min']:.1f} ans")
print(f"  Max: {age_stats['max']:.1f} ans")

# Analyse de l'expérience professionnelle
emp_stats = df_analysis["EMPLOYMENT_YEARS"].describe()
print(f"\nStatistiques d'expérience professionnelle:")
print(f"  Moyenne: {emp_stats['mean']:.1f} ans")
print(f"  Médiane: {emp_stats['50%']:.1f} ans")
print(f"  Min: {emp_stats['min']:.1f} ans")
print(f"  Max: {emp_stats['max']:.1f} ans")

# Visualisation de l'âge vs taux de défaut
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Distribution de l'âge par classe
age_data = [
    df_analysis[df_analysis["TARGET"] == 0]["AGE_YEARS"],
    df_analysis[df_analysis["TARGET"] == 1]["AGE_YEARS"],
]
axes[0].boxplot(age_data, labels=["Bons clients", "Défauts"])
axes[0].set_title("Distribution de l'âge par classe")
axes[0].set_ylabel("Âge (années)")

# Distribution de l'expérience par classe
emp_data = [
    df_analysis[df_analysis["TARGET"] == 0]["EMPLOYMENT_YEARS"],
    df_analysis[df_analysis["TARGET"] == 1]["EMPLOYMENT_YEARS"],
]
axes[1].boxplot(emp_data, labels=["Bons clients", "Défauts"])
axes[1].set_title("Distribution de l'expérience par classe")
axes[1].set_ylabel("Expérience (années)")

plt.tight_layout()
plt.savefig("reports/temporal_analysis.png", dpi=300, bbox_inches="tight")
print("Graphique d'analyse temporelle généré et sauvegardé")

# =============================================================================
# Insights et recommandations
# =============================================================================
print("\nINSIGHTS ET RECOMMANDATIONS")
print("=" * 40)

# Calcul de quelques métriques intéressantes
total_missing = df_train.isnull().sum().sum()
total_cells = len(df_train) * len(df_train.columns)
missing_percentage = (total_missing / total_cells) * 100

insights = [
    f"**VARIABLE CIBLE**:",
    f"   - Déséquilibre important: {target_counts[0]/target_counts[1]:.1f}:1",
    f"   - Stratégies de rééquilibrage nécessaires (SMOTE, weights)",
    f"",
    f"**QUALITÉ DES DONNÉES**:",
    f"   - {missing_percentage:.1f}% de valeurs manquantes au total",
    f"   - {len(missing_df)} colonnes avec valeurs manquantes",
    f"   - Stratégie d'imputation nécessaire",
    f"",
    f"**FEATURES NUMÉRIQUES**:",
    f"   - Variables temporelles à transformer (âge, expérience)",
    f"   - Outliers présents dans les montants financiers",
    f"   - Normalisation/standardisation recommandée",
    f"",
    f"**FEATURES CATÉGORIELLES**:",
    f"   - Encodage nécessaire (Label/OneHot)",
    f"   - Certaines catégories rares à regrouper",
    f"   - Potentiel pour feature engineering",
    f"",
    f"**RECOMMANDATIONS POUR LA MODÉLISATION**:",
    f"   - Utiliser des modèles robustes aux valeurs manquantes (LightGBM, XGBoost)",
    f"   - Implémenter un score métier avec coûts asymétriques",
    f"   - Cross-validation stratifiée obligatoire",
    f"   - Feature selection pour réduire la dimensionnalité",
    f"",
    f"**PROCHAINES ÉTAPES**:",
    f"   - Feature engineering avancé",
    f"   - Intégration des données auxiliaires (bureau, previous_application)",
    f"   - Test de différents algorithmes",
    f"   - Optimisation des hyperparamètres",
]

for insight in insights:
    print(insight)

# =============================================================================
# Sauvegarde des résultats
# =============================================================================
print("\nSAUVEGARDE DES RÉSULTATS")
print("=" * 40)

# Créer un résumé des découvertes
summary = {
    "execution_date": datetime.now().isoformat(),
    "dataset_shape": df_train.shape,
    "target_distribution": target_counts.to_dict(),
    "missing_values_count": len(missing_df),
    "top_missing_features": missing_df.head(10)["Colonne"].tolist(),
    "numeric_features_count": len(existing_numeric),
    "categorical_features_count": len(existing_categorical),
    "outliers_detected": {k: v["count"] for k, v in outliers_info.items()},
    "age_statistics": {
        "mean": float(age_stats["mean"]), # type: ignore
        "median": float(age_stats["50%"]), # type: ignore
        "min": float(age_stats["min"]), # type: ignore
        "max": float(age_stats["max"]), # type: ignore
    },
    "employment_statistics": {
        "mean": float(emp_stats["mean"]), # type: ignore
        "median": float(emp_stats["50%"]), # type: ignore
        "min": float(emp_stats["min"]), # type: ignore
        "max": float(emp_stats["max"]), # type: ignore
    },
    "top_correlations_with_target": (
        target_corr.to_dict() if "TARGET" in corr_matrix.columns else {} # type: ignore
    ),
    "recommendations": [
        "Gérer le déséquilibre des classes",
        "Traiter les valeurs manquantes",
        "Transformer les variables temporelles",
        "Encoder les variables catégorielles",
        "Gérer les outliers",
        "Implémenter un score métier",
    ],
}

# Sauvegarder en JSON
with open("reports/data_exploration_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("Résumé sauvegardé dans reports/data_exploration_summary.json")

# Affichage final
print("\nEXPLORATION TERMINÉE!")
print("=" * 40)
print("Prochaines étapes:")
print("   1. Feature engineering (notebook 02)")
print("   2. Preprocessing des données")
print("   3. Entraînement des modèles")
print("   4. Évaluation et optimisation")
print(f"\nFichiers générés dans le dossier reports/:")
print("   - Graphiques HTML interactifs")
print("   - Graphiques PNG statiques")
print("   - Résumé JSON des analyses")
