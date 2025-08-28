"""
Application principale MLOps Credit Scoring - Version Épurée
Interface Streamlit avec design moderne OpenClassrooms
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import sys
import numpy as np

# Ajouter le chemin src pour les imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Imports UI
from ui.theme import apply_theme
from ui.layout import render_main_layout
from ui.components import (
    ui_section_title,
    ui_subsection_title,
    ui_metric_card,
    ui_status_badge,
    ui_modern_gauge,
    ui_modern_chart,
    ui_empty_state,
    ui_info_box,
    ui_container,
)

# Configuration de la page
st.set_page_config(
    page_title="MLOps Credit Scoring",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
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


def predict_score(client_data, model_data):
    """Effectue une prédiction de score"""
    try:
        model = model_data["model"]
        threshold = model_data.get("threshold", 0.5)

        # Conversion en DataFrame
        df = pd.DataFrame([client_data])

        # Prédiction
        probabilities = model.predict_proba(df)
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

        return {
            "probability": probability,
            "decision": decision,
            "risk_level": risk_level,
            "threshold": threshold,
        }
    except Exception as e:
        st.error(f"Erreur prédiction: {e}")
        return None


def render_prediction_tab(model_data):
    """Onglet de prédiction individuelle"""
    ui_section_title(
        "Prédiction Individuelle", "Analysez le profil de risque d'un client"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        ui_subsection_title("Informations Client")

        # Informations de base
        col1a, col1b = st.columns(2)
        with col1a:
            gender = st.selectbox("Genre", ["M", "F"], key="gender")
            own_car = st.selectbox("Possède une voiture", ["Y", "N"], key="own_car")
            children = st.number_input(
                "Nombre d'enfants", min_value=0, max_value=20, value=0, key="children"
            )

        with col1b:
            own_realty = st.selectbox(
                "Possède un bien immobilier", ["Y", "N"], key="own_realty"
            )
            income = st.number_input(
                "Revenus annuels (€)",
                min_value=0,
                value=150000,
                step=1000,
                key="income",
            )
            credit = st.number_input(
                "Montant du crédit (€)",
                min_value=0,
                value=300000,
                step=1000,
                key="credit",
            )

        # Informations financières
        ui_subsection_title("Détails Financiers")
        col1c, col1d = st.columns(2)
        with col1c:
            annuity = st.number_input(
                "Annuité (€)", min_value=0, value=20000, step=100, key="annuity"
            )
            income_type = st.selectbox(
                "Type de revenus",
                [
                    "Working",
                    "Commercial associate",
                    "Pensioner",
                    "State servant",
                    "Student",
                ],
                key="income_type",
            )

        with col1d:
            goods_price = st.number_input(
                "Prix du bien (€)",
                min_value=0,
                value=280000,
                step=1000,
                key="goods_price",
            )
            education = st.selectbox(
                "Niveau d'éducation",
                [
                    "Secondary / secondary special",
                    "Higher education",
                    "Incomplete higher",
                    "Lower secondary",
                    "Academic degree",
                ],
                key="education",
            )

        # Informations complémentaires
        ui_subsection_title("Informations Complémentaires")
        col1e, col1f = st.columns(2)
        with col1e:
            family_status = st.selectbox(
                "Situation familiale",
                [
                    "Single / not married",
                    "Married",
                    "Civil marriage",
                    "Widow",
                    "Separated",
                ],
                key="family_status",
            )

        with col1f:
            housing_type = st.selectbox(
                "Type de logement",
                [
                    "House / apartment",
                    "Municipal apartment",
                    "With parents",
                    "Co-op apartment",
                    "Rented apartment",
                    "Office apartment",
                ],
                key="housing_type",
            )

        # Bouton de prédiction
        st.markdown("---")
        if st.button(
            "🔮 Analyser le Dossier", type="primary", use_container_width=True
        ):
            client_data = {
                "CODE_GENDER": gender,
                "FLAG_OWN_CAR": own_car,
                "FLAG_OWN_REALTY": own_realty,
                "CNT_CHILDREN": children,
                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity,
                "AMT_GOODS_PRICE": goods_price,
                "NAME_INCOME_TYPE": income_type,
                "NAME_EDUCATION_TYPE": education,
                "NAME_FAMILY_STATUS": family_status,
                "NAME_HOUSING_TYPE": housing_type,
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
                ui_metric_card("Probabilité de Défaut", f"{result['probability']:.1%}")
                ui_metric_card("Niveau de Risque", result["risk_level"])

            with col2b:
                ui_metric_card("Seuil Optimal", f"{result['threshold']:.1%}")
                ui_metric_card("Décision", result["decision"])

            # Badge de décision
            st.markdown("---")
            decision_type = "success" if result["decision"] == "ACCORDÉ" else "danger"
            ui_status_badge(f"CRÉDIT {result['decision']}", decision_type)

            # Graphique de risque
            st.markdown("### Visualisation du Risque")
            fig = ui_modern_gauge(
                result["probability"] * 100,
                "Risque de Défaut (%)",
                100,
                result["threshold"] * 100,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            ui_empty_state(
                "Prêt pour l'Analyse",
                "Remplissez le formulaire et cliquez sur 'Analyser le Dossier'",
                "🔮",
            )


def render_batch_analysis_tab(model_data):
    """Interface d'analyse en lot"""
    ui_section_title("Analyse en Lot", "Analysez plusieurs dossiers simultanément")

    def render_batch_content():
        # Template CSV
        ui_subsection_title("Télécharger le Template", "📄")

        if st.button("📥 Télécharger le template CSV", use_container_width=True):
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
                label="📥 Télécharger template.csv",
                data=csv,
                file_name="template_clients.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Upload de fichier
        ui_subsection_title("Télécharger vos Données", "📂")
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
                ui_subsection_title("Aperçu des Données", "👀")
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
            data=pd.DataFrame({"decision": decision_counts.index, "count": decision_counts.values}),
            x="decision",
            y="count",
            chart_type="bar",
            title="Distribution des Décisions",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Distribution des risques
        fig = ui_modern_chart(
            results_df,
            x="probability",
            chart_type="histogram",
            title="Distribution des Probabilités",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tableau des résultats
    ui_subsection_title("Détail des Résultats")
    st.dataframe(results_df, use_container_width=True)

    # Export
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Télécharger les résultats",
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
            ui_subsection_title("Prédictions Récentes", "📋")

            for i, entry in enumerate(reversed(st.session_state.history[-10:])):
                with st.expander(
                    f"Prédiction {len(st.session_state.history) - i} - {entry['timestamp']}",
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
                        export_data.append(
                            {
                                "timestamp": entry["timestamp"],
                                **entry["client_data"],
                                **entry["result"],
                            }
                        )

                    export_df = pd.DataFrame(export_data)
                    csv_export = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger l'historique",
                        data=csv_export,
                        file_name=f'historique_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime="text/csv",
                    )

        ui_container(render_history_content)
    else:
        ui_empty_state(
            "Aucune Prédiction",
            "Effectuez votre première prédiction dans l'onglet 'Prédiction'",
            "📈",
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

            # Graphique d'importance
            ui_subsection_title("Top 20 Features par Importance", "📊")
            top_features = feature_importance_df.head(20)

            fig = ui_modern_chart(
                top_features,
                chart_type="bar",
                x="importance",
                y="feature",
                orientation="h",
                title="Importance des Features",
                color="importance",
                color_continuous_scale=[
                    "#EF4444",
                    "#F59E0B",
                    "#10B981",
                    "#06B6D4",
                    "#7451F8",
                ],
            )
            st.plotly_chart(fig, use_container_width=True)

            # Recherche et tableau
            ui_subsection_title("Analyse Détaillée", "🔍")

            search_term = st.text_input(
                "Rechercher une feature", placeholder="Tapez le nom d'une feature..."
            )

            if search_term:
                filtered_df = feature_importance_df[
                    feature_importance_df["feature"].str.contains(
                        search_term, case=False, na=False
                    )
                ]
            else:
                filtered_df = feature_importance_df

            st.dataframe(filtered_df, use_container_width=True)

        ui_container(render_features_content)
    else:
        ui_empty_state(
            "Données Non Disponibles",
            "Les données d'importance des features ne sont pas encore chargées.",
            "📊",
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
                    ui_empty_state(title, "Rapport non disponible", "📊")

    ui_container(render_reports_content)


def render_main_content(model_data):
    """Contenu principal avec onglets"""
    # Navigation par onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🔮 Prédiction",
            "Analyse Lot",
            "Historique",
            "Features",
            "Rapports",
        ]
    )

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
    """Fonction principale de l'application"""
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

    # Rendu du layout principal
    render_main_layout(lambda: render_main_content(model_data), model_data)


if __name__ == "__main__":
    main()
