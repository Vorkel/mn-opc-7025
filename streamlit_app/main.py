"""
Application principale MLOps Credit Scoring

"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from feature_mapping import (
    get_feature_category,
    get_feature_description,
    get_readable_feature_name,
)

# Ajouter le chemin src pour les imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import du mapping des features


# Fonctions UI simplifiées pour remplacer les imports manquants
def apply_theme():
    """Applique le thème personnalisé"""
    st.markdown(
        """
        <style>
        .main { background-color: #f8f9fa; }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 5px;
        }
        .metric-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ui_section_title(title, subtitle=""):
    """Titre de section"""
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"*{subtitle}*")


def ui_subsection_title(title, icon=""):
    """Sous-titre de section"""
    st.markdown(f"### {icon} {title}")


def ui_metric_card(title, value):
    """Carte de métrique"""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(title, value)
    with col2:
        st.markdown("")


def ui_status_badge(text, status_type):
    """Badge de statut"""
    color = "green" if status_type == "success" else "red"
    st.markdown(
        f"<span style='color: {color}; font-weight: bold;'>{text}</span>",
        unsafe_allow_html=True,
    )


def ui_modern_gauge(value, min_val=0, max_val=1, title="Gauge"):
    """Jauge moderne"""
    try:
        if (
            isinstance(value, (int, float))
            and isinstance(min_val, (int, float))
            and isinstance(max_val, (int, float))
        ):
            percentage = (value - min_val) / (max_val - min_val) * 100
            st.progress(percentage / 100)
            st.markdown(f"**{title}**: {value:.1%}")
        else:
            st.markdown(f"**{title}**: {value}")
    except BaseException:
        st.markdown(f"**{title}**: {value}")


def ui_modern_chart(
    data,
    chart_type="bar",
    x=None,
    y=None,
    orientation="v",
    title="",
    color=None,
    color_continuous_scale=None,
):
    """Graphique moderne"""
    if chart_type == "bar":
        fig = px.bar(
            data,
            x=x,
            y=y,
            orientation=orientation,
            title=title,
            color=color,
            color_continuous_scale=color_continuous_scale,
        )
    else:
        fig = px.line(data, x=x, y=y, title=title)
    return fig


def ui_empty_state(title, message, icon=""):
    """État vide"""
    st.info(f"{icon} {title}: {message}")


def ui_info_box(message, message_type="info"):
    """Boîte d'information"""
    if message_type == "danger":
        st.error(message)
    elif message_type == "success":
        st.success(message)
    else:
        st.info(message)


def ui_container(func):
    """Conteneur"""
    func()


# Configuration de la page
st.set_page_config(
    page_title="Dashboard Credit Scoring - Prêt à Dépenser",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Chemins des fichiers
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"


def init_session_state():
    """Initialise les variables de session"""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_prediction" not in st.session_state:
        st.session_state.current_prediction = None


@st.cache_resource
def load_model(force_reload=False):
    """Charge le modèle entraîné"""
    try:
        model_paths = [
            MODELS_DIR / "best_credit_model.pkl",
            MODELS_DIR / "best_model.pkl",
            MODELS_DIR / "model.pkl",
            BASE_DIR / "model.pkl",
        ]

        for model_path in model_paths:
            if model_path.exists():
                model_data = joblib.load(model_path)
                return {
                    "model": model_data.get("model"),
                    "threshold": model_data.get("threshold", 0.5),
                    "scaler": model_data.get("scaler"),
                    "feature_names": model_data.get("feature_names", []),
                    "loaded_from": str(model_path),
                }
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None


@st.cache_data
def load_feature_importance():
    """Charge l'importance des features"""
    try:
        importance_path = DATA_DIR / "processed" / "feature_importance.csv"
        if importance_path.exists():
            return pd.read_csv(importance_path)
        return None
    except Exception:
        return None


def apply_feature_engineering(df):
    """Applique le même feature engineering que lors de l'entraînement"""
    try:
        df_engineered = df.copy()

        # Encodage des variables catégorielles
        categorical_mappings = {
            "CODE_GENDER": {"Homme": 1, "Femme": 0, "M": 1, "F": 0},
            "FLAG_OWN_CAR": {"Oui": 1, "Non": 0, "Y": 1, "N": 0},
            "FLAG_OWN_REALTY": {"Oui": 1, "Non": 0, "Y": 1, "N": 0},
            "NAME_CONTRACT_TYPE": {"Cash loans": 0, "Revolving loans": 1},
            "NAME_TYPE_SUITE": {
                "Unaccompanied": 0,
                "Family": 1,
                "Spouse, partner": 2,
                "Children": 3,
                "Other_B": 4,
                "Other_A": 5,
                "Group of people": 6,
            },
            "NAME_INCOME_TYPE": {
                "Salarié": 0,
                "Associé commercial": 1,
                "Retraité": 2,
                "Fonctionnaire": 3,
                "Étudiant": 5,
                "Working": 0,
                "Commercial associate": 1,
                "Pensioner": 2,
                "State servant": 3,
                "Student": 5,
            },
            "NAME_EDUCATION_TYPE": {
                "Secondaire": 0,
                "Supérieur": 1,
                "Supérieur incomplet": 2,
                "Secondaire inférieur": 3,
                "Diplôme universitaire": 4,
                "Secondary / secondary special": 0,
                "Higher education": 1,
                "Incomplete higher": 2,
                "Lower secondary": 3,
                "Academic degree": 4,
            },
            "NAME_FAMILY_STATUS": {
                "Célibataire": 2,
                "Marié": 1,
                "Union civile": 0,
                "Veuf/Veuve": 4,
                "Séparé": 3,
                "Single / not married": 2,
                "Married": 1,
                "Civil marriage": 0,
                "Widow": 4,
                "Separated": 3,
            },
            "NAME_HOUSING_TYPE": {
                "Maison/Appartement": 0,
                "Appartement municipal": 2,
                "Chez les parents": 1,
                "Appartement coopératif": 5,
                "Appartement loué": 3,
                "Appartement de fonction": 4,
                "House / apartment": 0,
                "Municipal apartment": 2,
                "With parents": 1,
                "Co-op apartment": 5,
                "Rented apartment": 3,
                "Office apartment": 4,
            },
            "OCCUPATION_TYPE": {
                "Laborers": 0,
                "Core staff": 1,
                "Accountants": 2,
                "Managers": 3,
                "Drivers": 4,
                "Sales staff": 5,
                "Cleaning staff": 6,
                "Cooking staff": 7,
                "Private service staff": 8,
                "Medicine staff": 9,
                "Security staff": 10,
                "High skill tech staff": 11,
                "Waiters/barmen staff": 12,
                "Low-skill Laborers": 13,
                "Realty agents": 14,
                "Secretaries": 15,
                "IT staff": 16,
                "HR staff": 17,
            },
            # Nouvelles mappings pour les features géographiques
            "ORGANIZATION_TYPE": {
                "Entreprise privée": 0,
                "Secteur public": 1,
                "Auto-entrepreneur": 2,
                "Association": 3,
                "ONG": 4,
                "Autre": 5,
                "Business Entity Type 1": 0,
                "Business Entity Type 2": 0,
                "Business Entity Type 3": 0,
                "Self-employed": 2,
                "Other": 5,
                "Medicine": 1,
                "Government": 1,
                "School": 1,
                "Trade: type 1": 0,
                "Trade: type 2": 0,
                "Trade: type 3": 0,
                "Trade: type 4": 0,
                "Trade: type 5": 0,
                "Trade: type 6": 0,
                "Trade: type 7": 0,
                "Industry: type 1": 0,
                "Industry: type 2": 0,
                "Industry: type 3": 0,
                "Industry: type 4": 0,
                "Industry: type 5": 0,
                "Industry: type 6": 0,
                "Industry: type 7": 0,
                "Industry: type 8": 0,
                "Industry: type 9": 0,
                "Industry: type 10": 0,
                "Industry: type 11": 0,
                "Industry: type 12": 0,
                "Industry: type 13": 0,
                "Transport: type 1": 0,
                "Transport: type 2": 0,
                "Transport: type 3": 0,
                "Transport: type 4": 0,
                "Cleaning": 0,
                "Security": 0,
                "Services": 0,
                "Hotel": 0,
                "Restaurant": 0,
                "Culture": 3,
                "Emergency": 1,
                "Military": 1,
                "Police": 1,
                "Postal": 1,
                "Realtor": 0,
                "Religion": 3,
                "University": 1,
                "XNA": 5,
            },
        }

        # Appliquer l'encodage
        for col, mapping in categorical_mappings.items():
            if col in df_engineered.columns:
                df_engineered[col] = df_engineered[col].map(mapping).fillna(0)

        # Préparer toutes les nouvelles features en une seule fois
        calculated_features = {}

        # Features temporelles (déjà définies dans notebooks/02_feature_engineering.py)
        if "DAYS_BIRTH" in df_engineered.columns:
            calculated_features["AGE_YEARS"] = -df_engineered["DAYS_BIRTH"] / 365.25

        if "DAYS_EMPLOYED" in df_engineered.columns:
            calculated_features["EMPLOYMENT_YEARS"] = (
                -df_engineered["DAYS_EMPLOYED"] / 365.25
            )

        # Ratios financiers (déjà optimisés dans le projet)
        if all(
            col in df_engineered.columns for col in ["AMT_CREDIT", "AMT_INCOME_TOTAL"]
        ):
            calculated_features["CREDIT_INCOME_RATIO"] = (
                df_engineered["AMT_CREDIT"] / df_engineered["AMT_INCOME_TOTAL"]
            )

        if all(
            col in df_engineered.columns for col in ["AMT_ANNUITY", "AMT_INCOME_TOTAL"]
        ):
            calculated_features["ANNUITY_INCOME_RATIO"] = (
                df_engineered["AMT_ANNUITY"] / df_engineered["AMT_INCOME_TOTAL"]
            )

        if all(
            col in df_engineered.columns for col in ["AMT_CREDIT", "AMT_GOODS_PRICE"]
        ):
            calculated_features["CREDIT_GOODS_RATIO"] = (
                df_engineered["AMT_CREDIT"] / df_engineered["AMT_GOODS_PRICE"]
            )

        if all(col in df_engineered.columns for col in ["AMT_ANNUITY", "AMT_CREDIT"]):
            calculated_features["ANNUITY_CREDIT_RATIO"] = (
                df_engineered["AMT_ANNUITY"] / df_engineered["AMT_CREDIT"]
            )

        # Indicateur de valeurs manquantes
        if "AMT_ANNUITY" in df_engineered.columns:
            calculated_features["AMT_ANNUITY_MISSING"] = (
                df_engineered["AMT_ANNUITY"].isna().astype(int)
            )

        # Encodage des variables FLAG (Y/N -> 1/0)
        flag_columns = [
            col
            for col in df_engineered.columns
            if col.startswith("FLAG_")
            and col not in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]
        ]
        for col in flag_columns:
            if col in df_engineered.columns:
                df_engineered[col] = (
                    df_engineered[col].map({"Y": 1, "N": 0, 1: 1, 0: 0}).fillna(0)
                )

        # Features d'agrégation
        contact_features = [
            col
            for col in df_engineered.columns
            if col.startswith("FLAG_")
            and ("PHONE" in col or "MOBIL" in col or "EMAIL" in col)
        ]
        if contact_features:
            calculated_features["CONTACT_SCORE"] = df_engineered[contact_features].sum(
                axis=1
            )

        # Encodage des variables REGION et ORGANIZATION
        region_org_columns = [
            col
            for col in df_engineered.columns
            if "REGION" in col or "ORGANIZATION" in col
        ]
        for col in region_org_columns:
            if col in df_engineered.columns and df_engineered[col].dtype == "object":
                # Encodage simple basé sur l'ordre alphabétique
                unique_values = df_engineered[col].unique()
                mapping = {val: idx for idx, val in enumerate(sorted(unique_values))}
                df_engineered[col] = df_engineered[col].map(mapping).fillna(0)

        # Features externes (EXT_SOURCE) - déjà optimisées
        external_features = [
            col for col in df_engineered.columns if "EXT_SOURCE" in col
        ]
        if external_features:
            calculated_features["EXT_SOURCES_MEAN"] = df_engineered[
                external_features
            ].mean(axis=1)
            calculated_features["EXT_SOURCES_MAX"] = df_engineered[
                external_features
            ].max(axis=1)
            calculated_features["EXT_SOURCES_MIN"] = df_engineered[
                external_features
            ].min(axis=1)
            calculated_features["EXT_SOURCES_STD"] = df_engineered[
                external_features
            ].std(axis=1)
            calculated_features["EXT_SOURCES_COUNT"] = df_engineered[
                external_features
            ].count(axis=1)

            # Interaction âge/sources externes
            if (
                "AGE_YEARS" in calculated_features
                and "EXT_SOURCES_MEAN" in calculated_features
            ):
                calculated_features["AGE_EXT_SOURCES_INTERACTION"] = (
                    calculated_features["AGE_YEARS"]
                    * calculated_features["EXT_SOURCES_MEAN"]
                )

        # Ajouter toutes les nouvelles features calculées
        if calculated_features:
            for feature_name, feature_values in calculated_features.items():
                df_engineered[feature_name] = feature_values

        # Liste des features attendues par le modèle (alignée avec le projet existant)
        expected_features = [
            "AGE_YEARS",
            "EMPLOYMENT_YEARS",
            "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO",
            "CREDIT_GOODS_RATIO",
            "ANNUITY_CREDIT_RATIO",
            "AMT_ANNUITY_MISSING",
            "CONTACT_SCORE",
            "EXT_SOURCES_MEAN",
            "EXT_SOURCES_MAX",
            "EXT_SOURCES_MIN",
            "EXT_SOURCES_STD",
            "EXT_SOURCES_COUNT",
            "AGE_EXT_SOURCES_INTERACTION",
        ]

        # Remplir les valeurs manquantes avec 0
        for feature in expected_features:
            if feature not in df_engineered.columns:
                df_engineered[feature] = 0
            else:
                df_engineered[feature] = df_engineered[feature].fillna(0)

        # Conversion finale : s'assurer que toutes les colonnes sont numériques
        for col in df_engineered.columns:
            if df_engineered[col].dtype == "object":
                try:
                    numeric_col = pd.to_numeric(df_engineered[col], errors="coerce")
                    df_engineered[col] = numeric_col.fillna(0)  # type: ignore
                except BaseException:
                    # Si impossible de convertir, remplacer par 0
                    df_engineered[col] = 0

        return df_engineered

    except Exception as e:
        st.error(f"Erreur feature engineering: {e}")
        return df


def analyze_client_data(history):
    """Analyse les données des clients pour créer des insights"""
    if not history or len(history) < 2:
        return None

    try:
        # Extraire les données des clients
        client_data_list = []
        for pred in history:
            if "client_data" in pred:
                client_data_list.append(pred["client_data"])

        if not client_data_list:
            return None

        # Créer un DataFrame pour l'analyse
        df_clients = pd.DataFrame(client_data_list)

        # Calculer des statistiques par segment
        insights = {}

        # Analyse par âge
        if "DAYS_BIRTH" in df_clients.columns:
            ages = -df_clients["DAYS_BIRTH"] / 365.25
            insights["age_stats"] = {
                "mean": ages.mean(),
                "std": ages.std(),
                "min": ages.min(),
                "max": ages.max(),
            }

        # Analyse par revenus
        if "AMT_INCOME_TOTAL" in df_clients.columns:
            insights["income_stats"] = {
                "mean": df_clients["AMT_INCOME_TOTAL"].mean(),
                "std": df_clients["AMT_INCOME_TOTAL"].std(),
                "min": df_clients["AMT_INCOME_TOTAL"].min(),
                "max": df_clients["AMT_INCOME_TOTAL"].max(),
            }

        # Analyse par montant de crédit
        if "AMT_CREDIT" in df_clients.columns:
            insights["credit_stats"] = {
                "mean": df_clients["AMT_CREDIT"].mean(),
                "std": df_clients["AMT_CREDIT"].std(),
                "min": df_clients["AMT_CREDIT"].min(),
                "max": df_clients["AMT_CREDIT"].max(),
            }

        return insights

    except Exception as e:
        st.error(f"Erreur lors de l'analyse des données clients: {e}")
        return None


def validate_business_rules(client_data):
    """Valide les règles métier avant prédiction"""
    try:
        # Récupération des valeurs
        income = client_data.get("AMT_INCOME_TOTAL", 0)
        credit_amount = client_data.get("AMT_CREDIT", 0)
        annuity = client_data.get("AMT_ANNUITY", 0)
        goods_price = client_data.get("AMT_GOODS_PRICE", 0)

        # Règles de validation métier
        errors = []

        # 1. Revenus minimum
        if income < 12000:  # 1000€/mois minimum
            errors.append("Revenus annuels insuffisants (minimum 12 000€)")

        # 2. Ratio crédit/revenus (maximum 5x)
        if income > 0 and credit_amount / income > 5:
            errors.append(
                "Montant du crédit trop élevé par rapport aux revenus (max 5x)"
            )

        # 3. Ratio annuité/revenus (maximum 33%)
        if income > 0 and annuity / income > 0.33:
            errors.append("Annuité trop élevée par rapport aux revenus (max 33%)")

        # 4. Montant du crédit maximum
        if credit_amount > 2000000:  # 2 millions max
            errors.append("Montant du crédit trop élevé (maximum 2 000 000€)")

        # 5. Prix du bien cohérent avec le crédit
        if goods_price > 0 and abs(credit_amount - goods_price) / goods_price > 0.2:
            errors.append("Montant du crédit incohérent avec le prix du bien")

        # 6. Revenus maximum (détection d'erreur de saisie)
        if income > 10000000:  # 10 millions max
            errors.append("Revenus annuels anormalement élevés")

        # 7. Nouvelles validations - Phase 1
        # Âge minimum et maximum
        age_years = client_data.get("DAYS_BIRTH", 0)
        if age_years != 0:
            age_years = -age_years / 365.25
            if age_years < 18:
                errors.append("Âge insuffisant (minimum 18 ans)")
            elif age_years > 85:
                errors.append("Âge trop élevé (maximum 85 ans)")

        # Expérience professionnelle cohérente avec l'âge
        employment_years = client_data.get("DAYS_EMPLOYED", 0)
        if employment_years != 0 and age_years > 0:
            employment_years = -employment_years / 365.25
            if employment_years > age_years - 18:
                errors.append("Expérience professionnelle incohérente avec l'âge")

        # Âge du véhicule cohérent
        car_age = client_data.get("OWN_CAR_AGE", 0)
        if car_age > 25:
            errors.append("Âge du véhicule trop élevé (maximum 25 ans)")

        if errors:
            return {"valid": False, "message": " | ".join(errors)}

        return {"valid": True, "message": "Validation OK"}

    except Exception as e:
        return {"valid": False, "message": f"Erreur de validation: {str(e)}"}


def predict_score(client_data, model_data):
    """Effectue une prédiction de score"""
    try:
        model = model_data["model"]
        threshold = model_data.get("threshold", 0.5)
        feature_names = model_data.get("feature_names", [])

        # Validation métier des données d'entrée
        validation_result = validate_business_rules(client_data)
        if not validation_result["valid"]:
            return {
                "probability": 1.0,  # Risque maximum
                "decision": "REFUSÉ",
                "risk_level": "Élevé",
                "threshold": threshold,
                "validation_error": validation_result["message"],
            }

        # Conversion en DataFrame
        df = pd.DataFrame([client_data])

        # Appliquer le feature engineering
        df_engineered = apply_feature_engineering(df)

        # S'assurer que toutes les features attendues sont présentes
        for feature in feature_names:
            if feature not in df_engineered.columns:
                df_engineered[feature] = 0  # Valeur par défaut

        # Sélectionner uniquement les features attendues par le modèle
        df_final = df_engineered[feature_names]

        # Prédiction
        probabilities = model.predict_proba(df_final)  # type: ignore
        probability = probabilities[0][1]  # Probabilité de défaut

        # Décision basée sur le seuil optimisé
        decision = "REFUSÉ" if probability > threshold else "ACCORDÉ"

        # Niveau de risque
        if probability < 0.3:
            risk_level = "Faible"
        elif probability < 0.6:
            risk_level = "Modéré"
        else:
            risk_level = "Élevé"

        result = {
            "probability": probability,
            "decision": decision,
            "risk_level": risk_level,
            "threshold": threshold,
        }

        # Sauvegarder dans l'historique
        if "history" not in st.session_state:
            st.session_state.history = []

        # Ajouter la prédiction à l'historique avec timestamp
        from datetime import datetime

        prediction_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "client_data": client_data,
            "result": result,
        }
        st.session_state.history.append(prediction_record)

        # Forcer la mise à jour des graphiques
        st.session_state.update_trigger = not st.session_state.get(
            "update_trigger", False
        )

        return result
    except Exception as e:
        st.error(f"Erreur prédiction: {e}")
        return None


def render_dashboard_overview(model_data):
    """Tableau de bord principal avec métriques et aperçu"""
    st.markdown("## Tableau de Bord - Vue d'ensemble")

    # ===== SECTION 1: MÉTRIQUES PRINCIPALES =====
    st.markdown("### Métriques Principales")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modèle Actif", model_data.get("model_name", "Random Forest"))
    with col2:
        st.metric("Features Utilisées", len(model_data.get("feature_names", [])))
    with col3:
        st.metric("Seuil Optimisé", f"{model_data.get('threshold', 0.5):.3f}")
    with col4:
        total_predictions = len(st.session_state.history)
        if total_predictions > 0:
            approved_count = sum(
                1
                for pred in st.session_state.history
                if pred["result"]["decision"] == "ACCORDÉ"
            )
            approval_rate = (approved_count / total_predictions) * 100
            st.metric("Taux d'Accord", f"{approval_rate:.1f}%")
        else:
            st.metric("Prédictions", "0")

    # Indicateur de mise à jour
    if st.session_state.history:
        last_update = st.session_state.history[-1]["timestamp"]
        st.info(f"Dernière mise à jour : {last_update}")

    st.markdown("---")

    # ===== SECTION 2: GRAPHIQUES DE SYNTHÈSE =====
    st.markdown("### Graphiques de Synthèse")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Répartition des Décisions")
        if st.session_state.history:
            # Utiliser la clé de mise à jour pour forcer le recalcul
            _ = st.session_state.get("update_trigger", False)

            decisions = [
                pred["result"]["decision"] for pred in st.session_state.history
            ]
            decision_counts = pd.Series(decisions).value_counts()

            # Ajout de pourcentages dans les labels
            total_decisions = len(decisions)
            labels_with_percent = []
            for decision, count in decision_counts.items():
                percentage = (count / total_decisions) * 100
                labels_with_percent.append(f"{decision} ({percentage:.1f}%)")

            fig = px.pie(
                values=decision_counts.values,
                names=labels_with_percent,
                title="Répartition des Décisions",
                color_discrete_sequence=["#10B981", "#EF4444"],
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True, key="pie_decisions")
        else:
            st.info("Aucune prédiction effectuée pour le moment")

    with col2:
        st.markdown("#### Niveaux de Risque")
        if st.session_state.history:
            # Utiliser la clé de mise à jour pour forcer le recalcul
            _ = st.session_state.get("update_trigger", False)

            risks = [pred["result"]["risk_level"] for pred in st.session_state.history]
            risk_counts = pd.Series(risks).value_counts()

            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Distribution des Niveaux de Risque",
                color=risk_counts.values,
                color_continuous_scale="RdYlGn_r",
                text=risk_counts.values,
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True, key="bar_risk_levels")
        else:
            st.info("Aucune donnée disponible")

    # ===== SECTION 3: STATISTIQUES AVANCÉES =====
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Statistiques Avancées")

        # Utiliser la clé de mise à jour pour forcer le recalcul
        _ = st.session_state.get("update_trigger", False)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Risque moyen
            avg_risk = np.mean(
                [pred["result"]["probability"] for pred in st.session_state.history]
            )
            st.metric("Risque Moyen", f"{avg_risk:.1%}")

        with col2:
            # Écart-type des risques
            risk_std = np.std(
                [pred["result"]["probability"] for pred in st.session_state.history]
            )
            st.metric("Écart-type Risque", f"{risk_std:.1%}")

        with col3:
            # Dernière prédiction
            last_prediction = st.session_state.history[-1]
            last_risk = last_prediction["result"]["probability"]
            st.metric("Dernier Risque", f"{last_risk:.1%}")

        with col4:
            # Nombre total de prédictions
            st.metric("Total Prédictions", len(st.session_state.history))

        # Bouton pour forcer la mise à jour
        if st.button("Actualiser les Graphiques", use_container_width=True):
            st.rerun()

        # ===== SECTION 4: ANALYSES AVANCÉES =====
        st.markdown("---")
        st.markdown("### Analyses Avancées")

        # 1. Évolution temporelle et distribution des probabilités
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Évolution Temporelle")
            if len(st.session_state.history) > 1:
                # Préparer les données temporelles
                timestamps = [pred["timestamp"] for pred in st.session_state.history]
                probabilities = [
                    pred["result"]["probability"] for pred in st.session_state.history
                ]

                # Créer un DataFrame pour l'analyse temporelle
                df_temp = pd.DataFrame({
                    "timestamp": timestamps,
                    "probability": probabilities,
                    "decision": [
                        pred["result"]["decision"] for pred in st.session_state.history
                    ],
                })
                df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
                df_temp = df_temp.sort_values("timestamp")

                # Graphique d'évolution temporelle
                fig = px.line(
                    df_temp,
                    x="timestamp",
                    y="probability",
                    color="decision",
                    title="Évolution du Risque dans le Temps",
                    labels={
                        "probability": "Probabilité de Défaut",
                        "timestamp": "Date",
                    },
                    color_discrete_map={"ACCORDÉ": "#10B981", "REFUSÉ": "#EF4444"},
                )
                fig.add_hline(
                    y=0.5,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Seuil de Décision",
                )
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(
                    fig, use_container_width=True, key="line_temporal_evolution"
                )
            else:
                st.info("Au moins 2 prédictions nécessaires pour l'analyse temporelle")

        with col2:
            st.markdown("#### Distribution des Probabilités")
            if st.session_state.history:
                probabilities = [
                    pred["result"]["probability"] for pred in st.session_state.history
                ]

                # Histogramme des probabilités
                fig = px.histogram(
                    x=probabilities,
                    nbins=10,
                    title="Distribution des Probabilités de Défaut",
                    labels={"x": "Probabilité de Défaut", "y": "Nombre de Clients"},
                    color_discrete_sequence=["#6366F1"],
                )
                fig.add_vline(
                    x=0.5, line_dash="dash", line_color="red", annotation_text="Seuil"
                )
                st.plotly_chart(
                    fig, use_container_width=True, key="histogram_probabilities"
                )

        # 2. Analyse des features clés et segmentation
        st.markdown("#### Analyse des Facteurs Clés")
        col1, col2 = st.columns(2)

        with col1:
            # Simuler une analyse des features importantes (basée sur SHAP)
            if st.session_state.history:
                # Créer des données simulées pour l'analyse des features
                feature_importance = {
                    "Revenus": 0.25,
                    "Âge": 0.20,
                    "Expérience": 0.15,
                    "Ratio Crédit/Revenus": 0.18,
                    "Type de Logement": 0.12,
                    "Secteur d'Activité": 0.10,
                }

                fig = px.bar(
                    x=list(feature_importance.values()),
                    y=list(feature_importance.keys()),
                    orientation="h",
                    title="Importance des Features (SHAP)",
                    labels={"x": "Importance", "y": "Features"},
                    color=list(feature_importance.values()),
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(
                    fig, use_container_width=True, key="bar_feature_importance"
                )

        with col2:
            # Analyse des segments de clients
            if st.session_state.history:
                # Créer des segments basés sur les probabilités
                probabilities = [
                    pred["result"]["probability"] for pred in st.session_state.history
                ]

                # Définir les segments
                segments = []
                for prob in probabilities:
                    if prob < 0.3:
                        segments.append("Faible Risque")
                    elif prob < 0.6:
                        segments.append("Risque Moyen")
                    else:
                        segments.append("Risque Élevé")

                segment_counts = pd.Series(segments).value_counts()

                fig = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title="Segmentation des Clients par Risque",
                    color_discrete_map={
                        "Faible Risque": "#10B981",
                        "Risque Moyen": "#F59E0B",
                        "Risque Élevé": "#EF4444",
                    },
                )
                fig.update_traces(textinfo="percent+label")
                st.plotly_chart(
                    fig, use_container_width=True, key="pie_client_segmentation"
                )

        # 3. Métriques de performance
        st.markdown("#### Métriques de Performance")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.session_state.history:
                # Calculer des métriques avancées
                recent_predictions = (
                    st.session_state.history[-10:]
                    if len(st.session_state.history) >= 10
                    else st.session_state.history
                )
                recent_avg = np.mean(
                    [pred["result"]["probability"] for pred in recent_predictions]
                )
                overall_avg = np.mean(
                    [pred["result"]["probability"] for pred in st.session_state.history]
                )

                # Détecter les tendances
                if recent_avg > overall_avg * 1.1:
                    trend = "Hausse"
                elif recent_avg < overall_avg * 0.9:
                    trend = "Baisse"
                else:
                    trend = "Stable"

                st.metric("Tendance Récente", trend, f"{recent_avg:.1%}")

        with col2:
            if st.session_state.history:
                # Calculer la stabilité des prédictions
                probabilities = [
                    pred["result"]["probability"] for pred in st.session_state.history
                ]
                stability = 1 - np.std(probabilities)  # Plus stable = plus proche de 1

                if stability > 0.8:
                    stability_status = "Très Stable"
                elif stability > 0.6:
                    stability_status = "Stable"
                else:
                    stability_status = "Variable"

                st.metric("Stabilité du Modèle", stability_status, f"{stability:.2f}")

        with col3:
            if st.session_state.history:
                # Calculer le ratio accord/refus
                decisions = [
                    pred["result"]["decision"] for pred in st.session_state.history
                ]
                accord_count = decisions.count("ACCORDÉ")
                refuse_count = decisions.count("REFUSÉ")

                if refuse_count > 0:
                    ratio = accord_count / refuse_count
                else:
                    ratio = float("inf")

                st.metric(
                    "Ratio Accord/Refus",
                    f"{ratio:.2f}",
                    f"{accord_count}/{refuse_count}",
                )

        # 4. Alertes et recommandations
        st.markdown("#### Alertes et Recommandations")

        if st.session_state.history:
            alerts = []
            recommendations = []

            # Analyser les tendances
            probabilities = [
                pred["result"]["probability"] for pred in st.session_state.history
            ]
            recent_avg = (
                np.mean(probabilities[-5:])
                if len(probabilities) >= 5
                else np.mean(probabilities)
            )
            overall_avg = np.mean(probabilities)

            if recent_avg > overall_avg * 1.2:
                alerts.append("Augmentation significative du risque moyen")
                recommendations.append("Analyser les nouveaux profils de clients")

            if np.std(probabilities) > 0.3:
                alerts.append("Variabilité élevée des prédictions")
                recommendations.append("Vérifier la cohérence des données d'entrée")

            decisions = [
                pred["result"]["decision"] for pred in st.session_state.history
            ]
            refuse_rate = decisions.count("REFUSÉ") / len(decisions)

            if refuse_rate > 0.7:
                alerts.append("Taux de refus élevé")
                recommendations.append("Revoir les critères d'évaluation")
            elif refuse_rate < 0.2:
                alerts.append("Taux de refus très faible")
                recommendations.append("Renforcer les contrôles de risque")

            # Afficher les alertes
            if alerts:
                st.warning("**Alertes détectées :**")
                for alert in alerts:
                    st.write(f"• {alert}")

            # Afficher les recommandations
            if recommendations:
                st.info("**Recommandations :**")
                for rec in recommendations:
                    st.write(f"• {rec}")

            if not alerts and not recommendations:
                st.success("Aucune alerte détectée - Modèle stable")

        # 5. Analyse des profils clients
        st.markdown("#### Analyse des Profils Clients")

        if st.session_state.history:
            # Analyser les données des clients
            insights = analyze_client_data(st.session_state.history)

            if insights:
                col1, col2, col3 = st.columns(3)

                with col1:
                    if "age_stats" in insights:
                        age_stats = insights["age_stats"]
                        st.metric("Âge Moyen", f"{age_stats['mean']:.1f} ans")
                        st.metric("Écart-type Âge", f"{age_stats['std']:.1f} ans")

                with col2:
                    if "income_stats" in insights:
                        income_stats = insights["income_stats"]
                        st.metric("Revenus Moyens", f"{income_stats['mean']:,.0f} €")
                        st.metric("Écart-type Revenus", f"{income_stats['std']:,.0f} €")

                with col3:
                    if "credit_stats" in insights:
                        credit_stats = insights["credit_stats"]
                        st.metric("Crédit Moyen", f"{credit_stats['mean']:,.0f} €")
                        st.metric("Écart-type Crédit", f"{credit_stats['std']:,.0f} €")

                # Graphique de corrélation âge/risque
                if len(st.session_state.history) > 5:
                    st.markdown("##### Corrélation Âge/Risque")

                    # Extraire les données pour l'analyse
                    ages = []
                    risks = []

                    for pred in st.session_state.history:
                        if (
                            "client_data" in pred
                            and "DAYS_BIRTH" in pred["client_data"]
                        ):
                            age = -pred["client_data"]["DAYS_BIRTH"] / 365.25
                            risk = pred["result"]["probability"]
                            ages.append(age)
                            risks.append(risk)

                    if ages and risks:
                        # Créer un graphique de dispersion
                        fig = px.scatter(
                            x=ages,
                            y=risks,
                            title="Corrélation Âge vs Risque de Défaut",
                            labels={"x": "Âge (années)", "y": "Probabilité de Défaut"},
                            color=risks,
                            color_continuous_scale="RdYlGn_r",
                        )
                        fig.add_hline(
                            y=0.5,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Seuil",
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key="scatter_age_risk_correlation",
                        )

                        # Calculer la corrélation
                        correlation = np.corrcoef(ages, risks)[0, 1]
                        st.metric("Corrélation Âge/Risque", f"{correlation:.3f}")
                else:
                    st.info(
                        "Au moins 2 prédictions nécessaires pour l'analyse temporelle"
                    )

    # Actions rapides
    st.markdown("---")
    st.markdown("Actions Rapides")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Nouvelle Prédiction", use_container_width=True):
            st.session_state.current_page = "Prédiction Individuelle"
            st.rerun()
    with col2:
        if st.button("Analyser un Lot", use_container_width=True):
            st.session_state.current_page = "Analyse de Lot"
            st.rerun()
    with col3:
        if st.button("Voir les Rapports", use_container_width=True):
            st.session_state.current_page = "Rapports et Métriques"
            st.rerun()


def render_prediction_tab(model_data):
    """Onglet de prédiction individuelle"""
    ui_section_title(
        "Prédiction Individuelle", "Analysez le profil de risque d'un client"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        ui_subsection_title("Informations Client")

        # 1. INFORMATIONS PERSONNELLES ET SOCIO-DÉMOGRAPHIQUES
        ui_subsection_title("1. Informations Personnelles et Socio-démographiques")
        col1a, col1b = st.columns(2)
        with col1a:
            gender = st.selectbox("Genre", ["Homme", "Femme"], key="gender")
            age_years = st.number_input(
                "Âge (années)", min_value=18, max_value=100, value=35, key="age_years"
            )
            family_status = st.selectbox(
                "Situation familiale",
                [
                    "Célibataire",
                    "Marié",
                    "Union civile",
                    "Veuf/Veuve",
                    "Séparé",
                ],
                key="family_status",
            )
            children = st.number_input(
                "Nombre de personnes à charge",
                min_value=0,
                max_value=20,
                value=0,
                key="children",
            )

        with col1b:
            education = st.selectbox(
                "Niveau d'éducation",
                [
                    "Secondaire",
                    "Supérieur",
                    "Supérieur incomplet",
                    "Secondaire inférieur",
                    "Diplôme universitaire",
                ],
                key="education",
            )
            address_years = st.number_input(
                "Ancienneté à l'adresse actuelle (années)",
                min_value=0,
                max_value=50,
                value=5,
                key="address_years",
            )
            housing_type = st.selectbox(
                "Type de logement",
                [
                    "Propriétaire",
                    "Locataire",
                    "Hébergé gratuitement",
                    "Appartement de fonction",
                    "Logement social",
                ],
                key="housing_type",
            )

        # 2. INFORMATIONS PROFESSIONNELLES ET REVENUS
        ui_subsection_title("2. Informations Professionnelles et Revenus")
        col1c, col1d = st.columns(2)
        with col1c:
            employment_status = st.selectbox(
                "Statut professionnel",
                [
                    "Salarié CDI",
                    "Salarié CDD",
                    "Indépendant",
                    "Fonctionnaire",
                    "Retraité",
                    "Étudiant",
                    "Chômeur",
                ],
                key="employment_status",
            )
            employment_years = st.number_input(
                "Ancienneté dans l'emploi actuel (années)",
                min_value=0,
                max_value=50,
                value=5,
                key="employment_years",
            )
            sector_activity = st.selectbox(
                "Secteur d'activité",
                [
                    "Industrie",
                    "Services",
                    "Commerce",
                    "Construction",
                    "Agriculture",
                    "Finance",
                    "Santé",
                    "Éducation",
                    "Administration",
                    "Autre",
                ],
                key="sector_activity",
            )

        with col1d:
            income_monthly = st.number_input(
                "Revenus nets mensuels (€)",
                min_value=0,
                value=5000,
                step=100,
                key="income_monthly",
                format="%d",
            )
            income_variable = st.number_input(
                "Revenus variables mensuels (€)",
                min_value=0,
                value=500,
                step=100,
                key="income_variable",
                format="%d",
            )
            other_income = st.number_input(
                "Autres sources de revenus (€/mois)",
                min_value=0,
                value=0,
                step=100,
                key="other_income",
                format="%d",
            )

        # 3. CHARGES ET ENDETTEMENT
        ui_subsection_title("3. Charges et Endettement")
        col1e, col1f = st.columns(2)
        with col1e:
            rent_mortgage = st.number_input(
                "Loyer ou mensualité crédit immobilier (€)",
                min_value=0,
                value=1200,
                step=50,
                key="rent_mortgage",
                format="%d",
            )
            other_credits = st.number_input(
                "Autres crédits en cours (€/mois)",
                min_value=0,
                value=300,
                step=50,
                key="other_credits",
                format="%d",
            )
            total_monthly_payments = st.number_input(
                "Total mensualités actuelles (€)",
                min_value=0,
                value=1500,
                step=50,
                key="total_monthly_payments",
                format="%d",
            )

        with col1f:
            debt_ratio = st.slider(
                "Taux d'endettement (%)",
                min_value=0,
                max_value=100,
                value=30,
                key="debt_ratio",
            )
            disposable_income = st.number_input(
                "Reste à vivre (€/mois)",
                min_value=0,
                value=2000,
                step=100,
                key="disposable_income",
                format="%d",
            )

        # 4. HISTORIQUE FINANCIER ET BANCAIRE
        ui_subsection_title("4. Historique Financier et Bancaire")
        col1g, col1h = st.columns(2)
        with col1g:
            bank_years = st.number_input(
                "Ancienneté bancaire (années)",
                min_value=0,
                max_value=50,
                value=10,
                key="bank_years",
            )
            payment_history = st.selectbox(
                "Historique de paiement",
                [
                    "Excellent",
                    "Bon",
                    "Moyen",
                    "Problématique",
                    "Défaut",
                ],
                key="payment_history",
            )
            overdraft_frequency = st.selectbox(
                "Découverts bancaires",
                [
                    "Jamais",
                    "Occasionnel",
                    "Régulier",
                    "Fréquent",
                ],
                key="overdraft_frequency",
            )

        with col1h:
            savings_amount = st.number_input(
                "Épargne disponible (€)",
                min_value=0,
                value=15000,
                step=1000,
                key="savings_amount",
                format="%d",
            )
            credit_bureau_score = st.selectbox(
                "Cotation Banque de France",
                [
                    "Aucun incident",
                    "Incidents mineurs",
                    "Incidents modérés",
                    "Incidents majeurs",
                    "Fichage",
                ],
                key="credit_bureau_score",
            )

        # 5. CARACTÉRISTIQUES DU CRÉDIT DEMANDÉ
        ui_subsection_title("5. Caractéristiques du Crédit Demandé")
        col1i, col1j = st.columns(2)
        with col1i:
            credit_amount = st.number_input(
                "Montant du crédit (€)",
                min_value=0,
                value=300000,
                step=1000,
                key="credit_amount",
                format="%d",
            )
            credit_duration = st.number_input(
                "Durée souhaitée (années)",
                min_value=1,
                max_value=30,
                value=20,
                key="credit_duration",
            )
            credit_purpose = st.selectbox(
                "Finalité du crédit",
                [
                    "Immobilier",
                    "Consommation",
                    "Regroupement de crédits",
                    "Automobile",
                    "Travaux",
                    "Professionnel",
                ],
                key="credit_purpose",
            )

        with col1j:
            personal_contribution = st.number_input(
                "Apport personnel (€)",
                min_value=0,
                value=50000,
                step=1000,
                key="personal_contribution",
                format="%d",
            )
            guarantee_type = st.selectbox(
                "Type de garantie",
                [
                    "Hypothèque",
                    "Caution",
                    "Nantissement",
                    "Assurance",
                    "Aucune",
                ],
                key="guarantee_type",
            )

        # 6. DONNÉES COMPORTEMENTALES
        ui_subsection_title("6. Données Comportementales")
        col1k, col1l = st.columns(2)
        with col1k:
            spending_habits = st.selectbox(
                "Habitudes de dépenses",
                [
                    "Économique",
                    "Modéré",
                    "Élevé",
                    "Luxueux",
                ],
                key="spending_habits",
            )
            income_stability = st.selectbox(
                "Régularité des revenus",
                [
                    "Très stable",
                    "Stable",
                    "Variable",
                    "Irrégulier",
                ],
                key="income_stability",
            )

        with col1l:
            balance_evolution = st.selectbox(
                "Évolution du solde bancaire",
                [
                    "En augmentation",
                    "Stable",
                    "En diminution",
                    "Variable",
                ],
                key="balance_evolution",
            )

        # 7. DONNÉES CONTEXTUELLES
        ui_subsection_title("7. Données Contextuelles")
        col1m, col1n = st.columns(2)
        with col1m:
            region_population = st.selectbox(
                "Densité de population",
                ["Faible", "Moyenne", "Élevée"],
                key="region_population",
            )
            unemployment_rate = st.selectbox(
                "Taux de chômage local",
                [
                    "Faible (< 5%)",
                    "Moyen (5-10%)",
                    "Élevé (10-15%)",
                    "Très élevé (> 15%)",
                ],
                key="unemployment_rate",
            )

        with col1n:
            real_estate_trend = st.selectbox(
                "Évolution prix immobilier local",
                [
                    "En hausse",
                    "Stable",
                    "En baisse",
                    "Variable",
                ],
                key="real_estate_trend",
            )

        # Bouton de prédiction
        st.markdown("---")
        if st.button("Analyser le Dossier", type="primary", use_container_width=True):
            client_data = {
                # 1. Informations personnelles et socio-démographiques
                "CODE_GENDER": {"Homme": "M", "Femme": "F"}[gender],
                "CNT_CHILDREN": children,
                "NAME_EDUCATION_TYPE": education,
                "NAME_FAMILY_STATUS": family_status,
                "NAME_HOUSING_TYPE": housing_type,
                "DAYS_BIRTH": -age_years * 365.25,
                "DAYS_REGISTRATION": -address_years * 365.25,
                # 2. Informations professionnelles et revenus
                "NAME_INCOME_TYPE": employment_status,
                "DAYS_EMPLOYED": -employment_years * 365.25,
                "AMT_INCOME_TOTAL": (
                    income_monthly * 12 + income_variable * 12 + other_income * 12
                ),
                # 3. Charges et endettement
                "AMT_ANNUITY": rent_mortgage + other_credits,
                "DEBT_RATIO": debt_ratio / 100,
                "DISPOSABLE_INCOME": disposable_income,
                # 4. Historique financier et bancaire
                "BANK_YEARS": bank_years,
                "PAYMENT_HISTORY": payment_history,
                "OVERDRAFT_FREQUENCY": overdraft_frequency,
                "SAVINGS_AMOUNT": savings_amount,
                "CREDIT_BUREAU_SCORE": credit_bureau_score,
                # 5. Caractéristiques du crédit demandé
                "AMT_CREDIT": credit_amount,
                "CREDIT_DURATION": credit_duration,
                "CREDIT_PURPOSE": credit_purpose,
                "PERSONAL_CONTRIBUTION": personal_contribution,
                "GUARANTEE_TYPE": guarantee_type,
                # 6. Données comportementales
                "SPENDING_HABITS": spending_habits,
                "INCOME_STABILITY": income_stability,
                "BALANCE_EVOLUTION": balance_evolution,
                # 7. Données contextuelles
                "REGION_POPULATION_RELATIVE": {
                    "Faible": 0.1,
                    "Moyenne": 0.5,
                    "Élevée": 0.9,
                }[region_population],
                "UNEMPLOYMENT_RATE": unemployment_rate,
                "REAL_ESTATE_TREND": real_estate_trend,
                # Features calculées automatiquement
                "REGION_RATING_CLIENT": 2,
                "SECTOR_ACTIVITY": sector_activity,
            }

            result = predict_score(client_data, model_data)

            if result:
                st.session_state.current_prediction = {
                    "client_data": client_data,
                    "result": result,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                # Ajouter à l'historique
                st.session_state.history.append(st.session_state.current_prediction)
                st.rerun()

    with col2:
        if st.session_state.current_prediction:
            result = st.session_state.current_prediction["result"]

            ui_subsection_title("Résultat de l'Analyse")

            # Métriques principales
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Probabilité de Défaut", f"{result['probability']:.1%}")
                st.metric("Niveau de Risque", result["risk_level"])
            with col2b:
                st.metric("Seuil Optimal", f"{result['threshold']:.1%}")
                st.metric("Décision", result["decision"])

            # Badge de décision
            st.markdown("---")
            if result["decision"] == "ACCORDÉ":
                st.success(f"CRÉDIT {result['decision']}")
            else:
                st.error(f"CRÉDIT {result['decision']}")

            # Affichage des erreurs de validation si présentes
            if "validation_error" in result:
                st.error(f"**Raison du refus :** {result['validation_error']}")

            # Graphique de risque avec jauge
            st.markdown("### Visualisation du Risque")
            risk_percentage = float(result["probability"] * 100)

            # Jauge avec barre de progression et couleur selon le risque
            if risk_percentage < 30:
                color = "green"
            elif risk_percentage < 60:
                color = "orange"
            else:
                color = "red"

            # Jauge principale avec style personnalisé
            st.markdown(
                f"""
            <div style="margin: 20px 0;">
                <div style="
                    background-color: #f0f0f0;
                    border-radius: 10px;
                    padding: 10px;
                ">
                    <div style="
                        background-color: {color};
                        height: 30px;
                        border-radius: 8px;
                        width: {risk_percentage}%;
                        transition: width 0.5s;
                    "></div>
                </div>
                <div style="
                    text-align: center;
                    margin-top: 10px;
                    font-size: 18px;
                    font-weight: bold;
                ">
                    Risque de Défaut: {risk_percentage:.1f}%
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Jauge circulaire supplémentaire
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Jauge Circulaire")
                # Création d'une jauge circulaire avec Plotly
                fig_gauge = px.pie(
                    values=[risk_percentage, 100 - risk_percentage],
                    names=["Risque", "Sécurité"],
                    title=f"Risque: {risk_percentage:.1f}%",
                    color_discrete_sequence=[color, "#e0e0e0"],
                )
                fig_gauge.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_risk")

            with col2:
                st.markdown("### Comparaison Risque vs Seuil")
                # Graphique en barres pour comparaison
                risk_data = pd.DataFrame({
                    "Métrique": ["Risque Client", "Seuil Optimal"],
                    "Valeur (%)": [risk_percentage, result["threshold"] * 100],
                })

                fig = px.bar(
                    risk_data,
                    x="Métrique",
                    y="Valeur (%)",
                    title="Comparaison Risque vs Seuil",
                    color="Métrique",
                    color_discrete_map={
                        "Risque Client": "#ff6b6b",
                        "Seuil Optimal": "#4ecdc4",
                    },
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(
                    fig, use_container_width=True, key="bar_risk_comparison"
                )
        else:
            ui_empty_state(
                "Prêt pour l'Analyse",
                "Remplissez le formulaire et cliquez sur 'Analyser le Dossier'",
            )


def render_batch_analysis_tab(model_data):
    """Interface d'analyse en lot"""
    ui_section_title("Analyse en Lot", "Analysez plusieurs dossiers simultanément")

    def render_batch_content():
        # Template CSV
        ui_subsection_title("Télécharger le Template")

        if st.button("Télécharger le template CSV", use_container_width=True):
            template_data = {
                "CODE_GENDER": ["M", "F"],
                "FLAG_OWN_CAR": ["Y", "N"],
                "FLAG_OWN_REALTY": ["Y", "Y"],
                "CNT_CHILDREN": [0, 2],
                "AMT_INCOME_TOTAL": [200000, 150000],
                "AMT_CREDIT": [400000, 250000],
                "AMT_ANNUITY": [25000, 18000],
                "AMT_GOODS_PRICE": [350000, 220000],
                "NAME_INCOME_TYPE": ["Working", "Working"],
                "NAME_EDUCATION_TYPE": [
                    "Secondary / secondary special",
                    "Higher education",
                ],
                "NAME_FAMILY_STATUS": ["Married", "Single / not married"],
                "NAME_HOUSING_TYPE": ["House / apartment", "House / apartment"],
            }

            template_df = pd.DataFrame(template_data)
            csv = template_df.to_csv(index=False)
            st.download_button(
                label="Télécharger template.csv",
                data=csv,
                file_name="template_clients.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Upload de fichier
        ui_subsection_title("Télécharger vos Données")
        uploaded_file = st.file_uploader(
            "Glissez-déposez votre fichier CSV ici",
            type=["csv"],
            help="Le fichier doit contenir toutes les colonnes requises",
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # Informations du fichier
                col1, col2, col3 = st.columns(3)
                with col1:
                    ui_metric_card("Nombre de Clients", f"{len(df):,}")
                with col2:
                    ui_metric_card("Colonnes", f"{len(df.columns)}")
                with col3:
                    size_kb = df.memory_usage(deep=True).sum() / 1024
                    ui_metric_card("Taille", f"{size_kb:.1f} KB")

                # Aperçu des données
                ui_subsection_title("Aperçu des Données")
                st.dataframe(df.head(), use_container_width=True)

                # Analyse en lot
                if st.button(
                    "Lancer l'Analyse en Lot",
                    type="primary",
                    use_container_width=True,
                ):
                    render_batch_processing(df, model_data)

            except Exception as e:
                ui_info_box(f"Erreur traitement fichier: {str(e)}", "danger")

    ui_container(render_batch_content)


def render_batch_processing(df, model_data):
    """Traitement en lot des prédictions"""
    ui_subsection_title("Traitement en Cours")

    progress_bar = st.progress(0)
    results = []

    for i, row in df.iterrows():
        client_data = row.to_dict()
        result = predict_score(client_data, model_data)

        if result:
            results.append({"client_id": i + 1, **client_data, **result})

        progress_bar.progress((i + 1) / len(df))

    if results:
        results_df = pd.DataFrame(results)
        render_batch_results(results_df)


def render_batch_results(results_df):
    """Affichage des résultats d'analyse en lot"""
    ui_subsection_title("Résultats de l'Analyse")

    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ui_metric_card("Total Clients", f"{len(results_df):,}")
    with col2:
        accordes = len(results_df[results_df["decision"] == "ACCORDÉ"])
        ui_metric_card("Taux d'Acceptation", f"{accordes/len(results_df)*100:.1f}%")
    with col3:
        avg_prob = results_df["probability"].mean()
        ui_metric_card("Risque Moyen", f"{avg_prob:.1%}")
    with col4:
        high_risk = len(results_df[results_df["probability"] > 0.6])
        ui_metric_card("Clients Haut Risque", f"{high_risk}")

    # Graphiques
    col1, col2 = st.columns(2)

    with col1:
        # Distribution des décisions
        decision_counts = results_df["decision"].value_counts()
        fig = ui_modern_chart(
            data=pd.DataFrame(
                {"decision": decision_counts.index, "count": decision_counts.values}
            ),
            x="decision",
            y="count",
            chart_type="bar",
            title="Distribution des Décisions",
        )
        st.plotly_chart(
            fig, use_container_width=True, key="batch_decisions_distribution"
        )

    with col2:
        # Distribution des risques
        fig = ui_modern_chart(
            results_df,
            x="probability",
            chart_type="histogram",
            title="Distribution des Probabilités",
        )
        st.plotly_chart(
            fig, use_container_width=True, key="batch_probabilities_distribution"
        )

    # Tableau des résultats
    ui_subsection_title("Détail des Résultats")
    st.dataframe(results_df, use_container_width=True)

    # Export
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Télécharger les résultats",
        data=csv,
        file_name=f'resultats_analyse_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime="text/csv",
        use_container_width=True,
    )


def render_history_tab():
    """Interface d'historique"""
    ui_section_title(
        "Historique des Prédictions", "Consultez l'historique de vos analyses"
    )

    if st.session_state.history:

        def render_history_content():
            # Métriques de l'historique
            history_results = [entry["result"] for entry in st.session_state.history]

            col1, col2, col3 = st.columns(3)
            with col1:
                ui_metric_card("Total Prédictions", f"{len(history_results):,}")
            with col2:
                accordes = sum(
                    1 for r in history_results if r.get("decision") == "ACCORDÉ"
                )
                ui_metric_card(
                    "Taux d'Acceptation", f"{accordes/len(history_results)*100:.1f}%"
                )
            with col3:
                prob_moyenne = sum(
                    r.get("probability", 0) for r in history_results
                ) / len(history_results)
                ui_metric_card("Probabilité Moyenne", f"{prob_moyenne:.2%}")

            # Liste des prédictions
            ui_subsection_title("Prédictions Récentes")

            for i, entry in enumerate(reversed(st.session_state.history[-10:])):
                with st.expander(
                    f"Prédiction {len(st.session_state.history) - i} -"
                    f" {entry['timestamp']}",
                    expanded=False,
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        result = entry["result"]
                        decision = result["decision"]

                        if decision == "ACCORDÉ":
                            ui_status_badge("CRÉDIT ACCORDÉ", "success")
                        else:
                            ui_status_badge("CRÉDIT REFUSÉ", "danger")

                        st.metric("Probabilité", f"{result['probability']:.2%}")
                        st.metric("Niveau de Risque", result["risk_level"])

                    with col2:
                        client_data = entry["client_data"]
                        st.write(f"**Genre**: {client_data['CODE_GENDER']}")
                        st.write(
                            f"**Revenus**: {client_data['AMT_INCOME_TOTAL']:,.0f} €"
                        )
                        st.write(f"**Crédit**: {client_data['AMT_CREDIT']:,.0f} €")
                        st.write(f"**Enfants**: {client_data['CNT_CHILDREN']}")

            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Vider l'Historique", use_container_width=True):
                    st.session_state.history = []
                    ui_info_box("Historique vidé!", "success")
                    st.rerun()

            with col2:
                if st.button("📥 Exporter l'Historique", use_container_width=True):
                    export_data = []
                    for entry in st.session_state.history:
                        export_data.append({
                            "timestamp": entry["timestamp"],
                            **entry["client_data"],
                            **entry["result"],
                        })

                    export_df = pd.DataFrame(export_data)
                    csv_export = export_df.to_csv(index=False)
                    st.download_button(
                        label="Télécharger l'historique",
                        data=csv_export,
                        file_name=(
                            "historique_predictions_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        ),
                        mime="text/csv",
                    )

        ui_container(render_history_content)
    else:
        ui_empty_state(
            "Aucune Prédiction",
            "Effectuez votre première prédiction dans l'onglet 'Prédiction'",
        )


def render_features_tab():
    """Interface d'analyse des features"""
    ui_section_title("Analyse des Features", "Comprenez les facteurs clés du modèle")

    feature_importance_df = load_feature_importance()

    if feature_importance_df is not None:

        def render_features_content():
            # Métriques sur les features
            col1, col2, col3 = st.columns(3)
            with col1:
                ui_metric_card("Total Features", f"{len(feature_importance_df):,}")
            with col2:
                top_importance = (
                    feature_importance_df["importance"].iloc[0]
                    if len(feature_importance_df) > 0
                    else 0
                )
                ui_metric_card("Importance Max", f"{top_importance:.4f}")
            with col3:
                avg_importance = feature_importance_df["importance"].mean()
                ui_metric_card("Importance Moyenne", f"{avg_importance:.4f}")

            # Graphique d'importance avec noms compréhensibles
            ui_subsection_title("Top 20 Variables par Importance")
            top_features = feature_importance_df.head(20).copy()

            # Ajouter les noms compréhensibles
            top_features["Nom Compréhensible"] = top_features["feature"].apply(
                get_readable_feature_name
            )
            top_features["Description"] = top_features["feature"].apply(
                get_feature_description
            )
            top_features["Catégorie"] = top_features["feature"].apply(
                get_feature_category
            )

            fig = px.bar(
                top_features,
                x="importance",
                y="Nom Compréhensible",
                orientation="h",
                title="Importance des Variables",
                color="importance",
                color_continuous_scale="Viridis",
                hover_data=["Description", "Catégorie"],
            )
            fig.update_layout(
                xaxis_title="Importance", yaxis_title="Variables", height=600
            )
            st.plotly_chart(
                fig, use_container_width=True, key="features_importance_chart"
            )

            # Recherche et tableau avec noms compréhensibles
            ui_subsection_title("Analyse Détaillée")

            search_term = st.text_input(
                "Rechercher une variable", placeholder="Tapez le nom d'une variable..."
            )

            # Créer un DataFrame avec les noms compréhensibles
            display_df = feature_importance_df.copy()
            display_df["Nom Compréhensible"] = display_df["feature"].apply(
                get_readable_feature_name
            )
            display_df["Description"] = display_df["feature"].apply(
                get_feature_description
            )
            display_df["Catégorie"] = display_df["feature"].apply(get_feature_category)

            # Réorganiser les colonnes
            display_df = display_df[[
                "Nom Compréhensible",
                "importance",
                "Description",
                "Catégorie",
                "feature",
            ]]

            if search_term:
                filtered_df = display_df[
                    display_df["Nom Compréhensible"]
                    .astype(str)
                    .str.contains(search_term, case=False, na=False)  # type: ignore
                    | display_df["feature"]
                    .astype(str)
                    .str.contains(search_term, case=False, na=False)  # type: ignore
                ]  # type: ignore
            else:
                filtered_df = display_df

            st.dataframe(filtered_df, use_container_width=True)

        ui_container(render_features_content)
    else:
        ui_empty_state(
            "Données Non Disponibles",
            "Les données d'importance des features ne sont pas encore chargées.",
        )


def render_reports_tab():
    """Interface des rapports"""
    ui_section_title("Rapports et Visualisations", "Explorez les analyses approfondies")

    def render_reports_content():
        # Grille des rapports
        report_files = [
            ("feature_importance.png", "Importance des Features"),
            ("correlation_matrix.png", "Matrice de Corrélation"),
            ("numeric_features_distribution.png", "Distribution Features Numériques"),
            ("temporal_analysis.png", "Analyse Temporelle"),
            ("threshold_analysis.png", "Analyse des Seuils"),
        ]

        # Affichage en grille
        cols = st.columns(2)

        for i, (filename, title) in enumerate(report_files):
            file_path = REPORTS_DIR / filename
            col = cols[i % 2]

            with col:
                if file_path.exists():
                    ui_subsection_title(title)
                    st.image(str(file_path), use_container_width=True)
                else:
                    ui_empty_state(title, "Rapport non disponible")

    ui_container(render_reports_content)


def render_main_content(model_data):
    """Contenu principal avec onglets"""
    # Navigation par onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Prédiction",
        "Analyse Lot",
        "Historique",
        "Features",
        "Rapports",
    ])

    with tab1:
        render_prediction_tab(model_data)

    with tab2:
        render_batch_analysis_tab(model_data)

    with tab3:
        render_history_tab()

    with tab4:
        render_features_tab()

    with tab5:
        render_reports_tab()


def main():
    """Fonction principale de l'application - Dashboard moderne"""
    # Appliquer le thème
    apply_theme()

    # Initialiser session state
    init_session_state()

    # Chargement des données
    with st.spinner("Chargement du système..."):
        model_data = load_model()

    # Vérification du modèle
    if model_data is None:
        ui_empty_state(
            "Modèle Non Disponible",
            "Impossible de charger le modèle de scoring. Vérifiez la configuration.",
            "⚠️",
        )
        return

    # Sidebar - Navigation moderne
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.selectbox(
            "Sélectionner une section",
            [
                "Tableau de Bord",
                "Prédiction Individuelle",
                "Analyse de Lot",
                "Historique des Prédictions",
                "Analyse des Features",
                "Rapports et Métriques",
            ],
        )

        st.markdown("---")
        st.markdown("Métriques Globales")
        if model_data:
            st.metric("Modèle", model_data.get("model_name", "Random Forest"))
            st.metric("Features", len(model_data.get("feature_names", [])))
            st.metric("Seuil", f"{model_data.get('threshold', 0.5):.3f}")

        st.markdown("---")
        st.markdown("### Informations")
        st.info(
            "Dashboard Credit Scoring - Prêt à Dépenser\n\nVersion: 1.0\nDernière mise"
            " à jour: Aujourd'hui"
        )

    # Titre principal
    st.title("Dashboard Credit Scoring")
    st.markdown("**Prêt à Dépenser** - Système MLOps de scoring crédit")

    # Navigation par page
    if page == "Tableau de Bord":
        render_dashboard_overview(model_data)
    elif page == "Prédiction Individuelle":
        render_prediction_tab(model_data)
    elif page == "Analyse de Lot":
        render_batch_analysis_tab(model_data)
    elif page == "Historique des Prédictions":
        render_history_tab()
    elif page == "Analyse des Features":
        render_features_tab()
    elif page == "Rapports et Métriques":
        render_reports_tab()


if __name__ == "__main__":
    main()
