# Notebook: 02_feature_engineering.py
# Feature Engineering pour Home Credit Default Risk

"""
OBJECTIFS DE CE NOTEBOOK :
- Créer de nouvelles features à partir des données existantes
- Transformer les variables pour améliorer les performances
- Gérer les valeurs manquantes de manière intelligente
- Préparer les données pour la modélisation
- Évaluer l'impact des nouvelles features
"""

# =============================================================================
# Configuration et imports
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Configuration
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration des graphiques
try:
    plt.style.use("seaborn-v0_8")
except OSError:
    plt.style.use("seaborn")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)

print("Feature Engineering pour Home Credit")
print("=" * 50)
print(f"Date d'exécution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# Chargement des données
# =============================================================================
print("\nChargement des données...")
print("=" * 30)

try:
    # Chargement des données principales
    df_train = pd.read_csv("data/raw/application_train.csv")
    df_test = pd.read_csv("data/raw/application_test.csv")

    print(f"Données chargées avec succès!")
    print(f"Train: {df_train.shape}")
    print(f"Test: {df_test.shape}")

    # Préparer les données de base
    train_ids = df_train["SK_ID_CURR"].copy()
    test_ids = df_test["SK_ID_CURR"].copy()
    target = df_train["TARGET"].copy()

    # Combiner train et test pour preprocessing uniforme
    df_all = pd.concat(
        [df_train.drop("TARGET", axis=1), df_test], axis=0, ignore_index=True
    )

    print(f"Dataset combiné: {df_all.shape}")

except Exception as e:
    print(f"Erreur lors du chargement: {e}")
    exit(1)

# =============================================================================
# Analyse de la qualité des données
# =============================================================================
print("\nANALYSE DE LA QUALITÉ DES DONNÉES")
print("=" * 40)

# Informations générales
print(f"Nombre d'observations: {len(df_all):,}")
print(f"Nombre de features: {len(df_all.columns):,}")
print(f"Taille mémoire: {df_all.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Types de données
print(f"\nTypes de données:")
dtype_counts = df_all.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"   {dtype}: {count} colonnes")

# Valeurs manquantes
missing_data = df_all.isnull().sum()
missing_pct = (missing_data / len(df_all)) * 100
total_missing = missing_data.sum()
total_cells = len(df_all) * len(df_all.columns)
missing_percentage = (total_missing / total_cells) * 100

print(f"\nValeurs manquantes: {total_missing:,} ({missing_percentage:.2f}%)")
print(f"Colonnes avec valeurs manquantes: {(missing_data > 0).sum()}")

# =============================================================================
# Feature Engineering - Variables temporelles
# =============================================================================
print("\nFEATURE ENGINEERING - VARIABLES TEMPORELLES")
print("=" * 50)

# Créer une copie pour le feature engineering
df_engineered = df_all.copy()
new_features = []

# Convertir les jours en années
df_engineered["AGE_YEARS"] = -df_engineered["DAYS_BIRTH"] / 365.25
df_engineered["EMPLOYMENT_YEARS"] = -df_engineered["DAYS_EMPLOYED"] / 365.25

# Nettoyer les valeurs aberrantes DAYS_EMPLOYED
df_engineered["DAYS_EMPLOYED_ABNORMAL"] = (
    df_engineered["DAYS_EMPLOYED"] == 365243
).astype(int)
df_engineered["DAYS_EMPLOYED"] = df_engineered["DAYS_EMPLOYED"].replace(365243, np.nan)
df_engineered["EMPLOYMENT_YEARS"] = -df_engineered["DAYS_EMPLOYED"] / 365.25

# Variables temporelles supplémentaires
df_engineered["YEARS_SINCE_REGISTRATION"] = -df_engineered["DAYS_REGISTRATION"] / 365.25
df_engineered["YEARS_SINCE_ID_PUBLISH"] = -df_engineered["DAYS_ID_PUBLISH"] / 365.25

# Groupes d'âge
df_engineered["AGE_GROUP"] = pd.cut(
    df_engineered["AGE_YEARS"],
    bins=[0, 25, 35, 45, 55, 65, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
)

# Groupes d'expérience
df_engineered["EMPLOYMENT_GROUP"] = pd.cut(
    df_engineered["EMPLOYMENT_YEARS"],
    bins=[-1, 0, 2, 5, 10, 20, 50],
    labels=["Unemployed", "0-2y", "2-5y", "5-10y", "10-20y", "20y+"],
)

# Ratios temporels
df_engineered["AGE_EMPLOYMENT_RATIO"] = df_engineered["AGE_YEARS"] / (
    df_engineered["EMPLOYMENT_YEARS"] + 1
)

temporal_features = [
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "YEARS_SINCE_REGISTRATION",
    "YEARS_SINCE_ID_PUBLISH",
    "AGE_GROUP",
    "EMPLOYMENT_GROUP",
    "AGE_EMPLOYMENT_RATIO",
    "DAYS_EMPLOYED_ABNORMAL",
]

new_features.extend(temporal_features)
print(f"{len(temporal_features)} features temporelles créées")

# =============================================================================
# Feature Engineering - Variables financières
# =============================================================================
print("\nFEATURE ENGINEERING - VARIABLES FINANCIÈRES")
print("=" * 50)

# Ratios principaux
df_engineered["CREDIT_INCOME_RATIO"] = (
    df_engineered["AMT_CREDIT"] / df_engineered["AMT_INCOME_TOTAL"]
)
df_engineered["ANNUITY_INCOME_RATIO"] = (
    df_engineered["AMT_ANNUITY"] / df_engineered["AMT_INCOME_TOTAL"]
)
df_engineered["CREDIT_GOODS_RATIO"] = (
    df_engineered["AMT_CREDIT"] / df_engineered["AMT_GOODS_PRICE"]
)
df_engineered["ANNUITY_CREDIT_RATIO"] = (
    df_engineered["AMT_ANNUITY"] / df_engineered["AMT_CREDIT"]
)

# Durée estimée du crédit
df_engineered["CREDIT_DURATION"] = (
    df_engineered["AMT_CREDIT"] / df_engineered["AMT_ANNUITY"]
)

# Revenus et crédits par personne
df_engineered["INCOME_PER_PERSON"] = (
    df_engineered["AMT_INCOME_TOTAL"] / df_engineered["CNT_FAM_MEMBERS"]
)
df_engineered["CREDIT_PER_PERSON"] = (
    df_engineered["AMT_CREDIT"] / df_engineered["CNT_FAM_MEMBERS"]
)

# Groupes de revenus
df_engineered["INCOME_GROUP"] = pd.cut(
    df_engineered["AMT_INCOME_TOTAL"],
    bins=[0, 100000, 200000, 300000, 500000, np.inf],
    labels=["Low", "Medium", "High", "Very High", "Ultra High"],
)

# Groupes de crédit
df_engineered["CREDIT_GROUP"] = pd.cut(
    df_engineered["AMT_CREDIT"],
    bins=[0, 200000, 500000, 1000000, 2000000, np.inf],
    labels=["Small", "Medium", "Large", "Very Large", "Ultra Large"],
)

# Indicateurs de richesse
df_engineered["OWNS_PROPERTY"] = (
    (df_engineered["FLAG_OWN_CAR"] == "Y") & (df_engineered["FLAG_OWN_REALTY"] == "Y")
).astype(int)

df_engineered["OWNS_NEITHER"] = (
    (df_engineered["FLAG_OWN_CAR"] == "N") & (df_engineered["FLAG_OWN_REALTY"] == "N")
).astype(int)

financial_features = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "ANNUITY_CREDIT_RATIO",
    "CREDIT_DURATION",
    "INCOME_PER_PERSON",
    "CREDIT_PER_PERSON",
    "INCOME_GROUP",
    "CREDIT_GROUP",
    "OWNS_PROPERTY",
    "OWNS_NEITHER",
]

new_features.extend(financial_features)
print(f"{len(financial_features)} features financières créées")

# =============================================================================
# Feature Engineering - Variables d'agrégation
# =============================================================================
print("\nFEATURE ENGINEERING - VARIABLES D'AGRÉGATION")
print("=" * 50)

# Scores de contact
contact_features = [
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
]

df_engineered["CONTACT_SCORE"] = df_engineered[contact_features].sum(axis=1)

# Scores de documents
doc_features = [
    col for col in df_engineered.columns if col.startswith("FLAG_DOCUMENT_")
]
if doc_features:
    df_engineered["DOCUMENT_SCORE"] = df_engineered[doc_features].sum(axis=1)

# Score de région normalisé
df_engineered["REGION_SCORE_NORMALIZED"] = 4 - df_engineered["REGION_RATING_CLIENT"]

# Features externes (EXT_SOURCE)
external_features = [col for col in df_engineered.columns if "EXT_SOURCE" in col]
if external_features:
    df_engineered["EXT_SOURCES_MEAN"] = df_engineered[external_features].mean(axis=1)
    df_engineered["EXT_SOURCES_MAX"] = df_engineered[external_features].max(axis=1)
    df_engineered["EXT_SOURCES_MIN"] = df_engineered[external_features].min(axis=1)
    df_engineered["EXT_SOURCES_STD"] = df_engineered[external_features].std(axis=1)
    df_engineered["EXT_SOURCES_COUNT"] = df_engineered[external_features].count(axis=1) # type: ignore

    # Interactions
    df_engineered["AGE_EXT_SOURCES_INTERACTION"] = (
        df_engineered["AGE_YEARS"] * df_engineered["EXT_SOURCES_MEAN"]
    )

    ext_features = [
        "EXT_SOURCES_MEAN",
        "EXT_SOURCES_MAX",
        "EXT_SOURCES_MIN",
        "EXT_SOURCES_STD",
        "EXT_SOURCES_COUNT",
        "AGE_EXT_SOURCES_INTERACTION",
    ]
else:
    ext_features = []

agg_features = [
    "CONTACT_SCORE",
    "DOCUMENT_SCORE",
    "REGION_SCORE_NORMALIZED",
] + ext_features
new_features.extend(agg_features)
print(f"{len(agg_features)} features d'agrégation créées")

# =============================================================================
# Gestion des valeurs manquantes
# =============================================================================
print("\nGESTION DES VALEURS MANQUANTES")
print("=" * 40)

# Créer des indicateurs pour les features importantes
important_features = [
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "CNT_FAM_MEMBERS",
    "DAYS_REGISTRATION",
]

missing_indicators = []
for feature in important_features:
    if feature in df_engineered.columns:
        indicator_name = f"{feature}_MISSING"
        df_engineered[indicator_name] = df_engineered[feature].isnull().astype(int)
        missing_indicators.append(indicator_name)

# Imputation par type
# Variables numériques - médiane
numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_engineered[col].isnull().sum() > 0:
        median_val = df_engineered[col].median()
        df_engineered[col] = df_engineered[col].fillna(median_val)

# Variables catégorielles - mode
categorical_cols = df_engineered.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    if df_engineered[col].isnull().sum() > 0:
        mode_val = df_engineered[col].mode()
        if len(mode_val) > 0:
            df_engineered[col] = df_engineered[col].fillna(mode_val[0]) # type: ignore
        else:
            df_engineered[col] = df_engineered[col].fillna("Unknown")

new_features.extend(missing_indicators)
print(f"Valeurs manquantes gérées + {len(missing_indicators)} indicateurs créés")

# =============================================================================
# Encodage des variables catégorielles
# =============================================================================
print("\nENCODAGE DES VARIABLES CATÉGORIELLES")
print("=" * 45)

from sklearn.preprocessing import LabelEncoder

# Dictionnaire pour les encodeurs
encoders = {}

# Variables binaires
binary_features = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
for feature in binary_features:
    if feature in df_engineered.columns:
        le = LabelEncoder()
        df_engineered[feature] = le.fit_transform(df_engineered[feature].astype(str))
        encoders[feature] = le

# Variables ordinales
ordinal_mappings = {
    "NAME_EDUCATION_TYPE": {
        "Lower secondary": 1,
        "Secondary / secondary special": 2,
        "Incomplete higher": 3,
        "Higher education": 4,
        "Academic degree": 5,
    }
}

for feature, mapping in ordinal_mappings.items():
    if feature in df_engineered.columns:
        df_engineered[feature] = df_engineered[feature].map(mapping).fillna(0) # type: ignore

# Variables catégorielles standard
categorical_features = [
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "ORGANIZATION_TYPE",
    "NAME_CONTRACT_TYPE",
]

for feature in categorical_features:
    if feature in df_engineered.columns:
        le = LabelEncoder()
        df_engineered[feature] = le.fit_transform(df_engineered[feature].astype(str))
        encoders[feature] = le

# Nouvelles variables catégorielles
new_categorical = ["AGE_GROUP", "EMPLOYMENT_GROUP", "INCOME_GROUP", "CREDIT_GROUP"]
for feature in new_categorical:
    if feature in df_engineered.columns:
        le = LabelEncoder()
        df_engineered[feature] = le.fit_transform(df_engineered[feature].astype(str))
        encoders[feature] = le

# Encoder toutes les variables catégorielles restantes
remaining_categorical = df_engineered.select_dtypes(include=["object"]).columns
for feature in remaining_categorical:
    if feature not in encoders:
        le = LabelEncoder()
        df_engineered[feature] = le.fit_transform(df_engineered[feature].astype(str))
        encoders[feature] = le

print(f"{len(encoders)} variables catégorielles encodées")

# =============================================================================
# Visualisation des nouvelles features
# =============================================================================
print("\nVISUALISATION DES NOUVELLES FEATURES")
print("=" * 45)

# Créer le dossier reports/ si nécessaire
reports_path = Path("reports")
reports_path.mkdir(exist_ok=True)

# Visualiser quelques nouvelles features importantes
new_numeric_features = [
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "INCOME_PER_PERSON",
    "CONTACT_SCORE",
]

# Filtrer les features existantes
existing_features = [f for f in new_numeric_features if f in df_engineered.columns]

if existing_features:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    for i, feature in enumerate(existing_features):
        if i < 6:  # Limiter à 6 graphiques
            # Limiter aux percentiles pour éviter les outliers
            data = df_engineered[feature].dropna()
            if len(data) > 0:
                p99 = data.quantile(0.99)
                data_filtered = data[data <= p99]

                data_filtered.hist(bins=50, ax=axes[i]) # type: ignore
                axes[i].set_title(f"Distribution de {feature}")
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel("Fréquence")
                axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/new_features_distribution.png", dpi=300, bbox_inches="tight")
    print("Graphique des nouvelles features sauvegardé")

# =============================================================================
# Évaluation des performances
# =============================================================================
print("\nÉVALUATION DES PERFORMANCES")
print("=" * 35)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Préparer les données
train_len = len(target)
df_train_engineered = df_engineered[:train_len].copy()
df_train_engineered["TARGET"] = target

# Préparer X et y
X = df_train_engineered.drop(["TARGET", "SK_ID_CURR"], axis=1, errors="ignore")
y = df_train_engineered["TARGET"]

# Diviser les données
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Entraîner Random Forest
print("Entraînement du modèle Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)

rf.fit(X_train, y_train)

# Évaluer
y_pred = rf.predict_proba(X_val)[:, 1]  # type: ignore
auc_score = roc_auc_score(y_val, y_pred)

# Importance des features
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": rf.feature_importances_}
).sort_values("importance", ascending=False)

print(f"Performance avec nouvelles features: AUC = {auc_score:.4f}")

# Visualiser l'importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
sns.barplot(data=top_features, y="feature", x="importance")
plt.title("Top 20 Features les plus importantes")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=300, bbox_inches="tight")
print("Graphique d'importance des features sauvegardé")

# Analyser les nouvelles features dans le top
new_features_in_top = feature_importance[
    feature_importance["feature"].isin(new_features)
].head(10)

print(f"\nTop 10 nouvelles features les plus importantes:")
print(new_features_in_top)

# =============================================================================
# Sauvegarde des données finales
# =============================================================================
print("\nSAUVEGARDE DES DONNÉES FINALES")
print("=" * 35)

# Obtenir les données finales
df_train_final = df_engineered[:train_len].copy()
df_test_final = df_engineered[train_len:].copy()

# Ajouter les identifiants
df_train_final["SK_ID_CURR"] = train_ids
df_test_final["SK_ID_CURR"] = test_ids
df_train_final["TARGET"] = target

# Créer le dossier processed
import os

os.makedirs("data/processed", exist_ok=True)

# Sauvegarder les données
df_train_final.to_csv("data/processed/train_processed.csv", index=False)
df_test_final.to_csv("data/processed/test_processed.csv", index=False)

# Sauvegarder les encodeurs
import joblib

joblib.dump(encoders, "data/processed/label_encoders.pkl")

# Sauvegarder l'importance des features
feature_importance.to_csv("data/processed/feature_importance.csv", index=False)

print(f"Données sauvegardées:")
print(f"   train_processed.csv: {df_train_final.shape}")
print(f"   test_processed.csv: {df_test_final.shape}")
print(f"   label_encoders.pkl: {len(encoders)} encodeurs")
print(f"   feature_importance.csv: {len(feature_importance)} features")

# =============================================================================
# Génération du rapport
# =============================================================================
print("\nGÉNÉRATION DU RAPPORT")
print("=" * 25)

# Créer un rapport complet
report = {
    "execution_date": datetime.now().isoformat(),
    "original_features": len(df_all.columns),
    "new_features_created": len(new_features),
    "total_features": df_engineered.shape[1] if hasattr(df_engineered, 'shape') else 0,
    "data_quality": {
        "train_shape": df_train_final.shape,
        "test_shape": df_test_final.shape,
        "missing_values_handled": True,
        "categorical_encoded": True,
    },
    "performance": {"enhanced_auc": auc_score},
    "feature_categories": {
        "temporal": [
            f
            for f in new_features
            if any(x in f for x in ["AGE", "EMPLOYMENT", "YEARS"])
        ],
        "financial": [
            f
            for f in new_features
            if any(x in f for x in ["RATIO", "INCOME", "CREDIT"])
        ],
        "aggregated": [
            f for f in new_features if any(x in f for x in ["SCORE", "MEAN", "COUNT"])
        ],
        "indicators": [
            f for f in new_features if any(x in f for x in ["MISSING", "ABNORMAL"])
        ],
    },
    "top_features": feature_importance.head(10).to_dict("records"),
}

# Sauvegarder le rapport
import json

with open("reports/feature_engineering_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"Rapport sauvegardé dans reports/feature_engineering_report.json")

# =============================================================================
# Résumé final
# =============================================================================
print("\nRÉSUMÉ DU FEATURE ENGINEERING")
print("=" * 50)

print(f"TRANSFORMATIONS RÉALISÉES:")
print(f"   Features originales: {report['original_features']}")
print(f"   Nouvelles features: {report['new_features_created']}")
print(f"   Features totales: {report['total_features']}")
print(f"   Performance AUC: {report['performance']['enhanced_auc']:.4f}")

print(f"\nCATÉGORIES DE FEATURES:")
for category, features in report["feature_categories"].items():
    print(f"   {category.upper()}: {len(features)} features")
    if len(features) > 0:
        print(f"      Exemples: {features[:3]}")

print(f"\nINSIGHTS GÉNÉRÉS:")
print(f"   Features temporelles optimisées (âge, expérience)")
print(f"   Ratios financiers créés (crédit/revenu, etc.)")
print(f"   Scores d'agrégation calculés")
print(f"   Valeurs manquantes gérées intelligemment")
print(f"   Variables catégorielles encodées")

print(f"\nPROCHAINES ÉTAPES:")
print(f"   1. Entraînement des modèles (notebook 03)")
print(f"   2. Optimisation des hyperparamètres")
print(f"   3. Analyse SHAP (notebook 04)")
print(f"   4. Déploiement API et monitoring")

print("\nFEATURE ENGINEERING TERMINÉ!")
print("=" * 50)
