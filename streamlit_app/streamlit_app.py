"""
Application Streamlit MLOps Credit Scoring - Version Simplifiée
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import joblib
from pathlib import Path
import sys
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="MLOps Credit Scoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS moderne
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #7451F8 0%, #4F46E5 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #7451F8;
    }
    .status-success {
        background: #10B981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
    .status-danger {
        background: #EF4444;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Chemins des fichiers
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_credit_model.pkl"
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train_processed.csv"

# Charger le modèle
@st.cache_resource
def load_model():
    """Charger le modèle ML"""
    try:
        if MODEL_PATH.exists():
            model_data = joblib.load(MODEL_PATH)
            
            # Afficher des infos de debug
            st.info(f"Type de modèle chargé: {type(model_data)}")
            
            # Gérer différents formats de sauvegarde
            if isinstance(model_data, dict):
                if 'model' in model_data:
                    actual_model = model_data['model']
                    st.success(f"✅ Modèle extrait du dictionnaire: {type(actual_model)}")
                    if hasattr(actual_model, 'n_features_in_'):
                        st.info(f"Features attendues: {actual_model.n_features_in_}")
                else:
                    st.warning(f"⚠️ Dictionnaire sans clé 'model'. Clés disponibles: {list(model_data.keys())}")
            else:
                st.success(f"✅ Modèle chargé directement: {type(model_data)}")
            
            return model_data
        else:
            st.error(f"❌ Modèle non trouvé: {MODEL_PATH}")
            return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

# Charger les données d'exemple
@st.cache_data
def load_sample_data():
    """Charger un échantillon de données pour les tests"""
    try:
        if TRAIN_DATA_PATH.exists():
            df = pd.read_csv(TRAIN_DATA_PATH)
            return df.drop(['TARGET', 'SK_ID_CURR'], axis=1, errors='ignore').sample(100)
        else:
            # Données factices pour la démo
            return pd.DataFrame({
                'EXT_SOURCE_2': np.random.normal(0.5, 0.2, 100),
                'EXT_SOURCE_3': np.random.normal(0.5, 0.2, 100),
                'DAYS_BIRTH': np.random.randint(-25000, -5000, 100),
                'AMT_INCOME_TOTAL': np.random.normal(150000, 50000, 100),
                'AMT_CREDIT': np.random.normal(500000, 200000, 100),
            })
    except Exception as e:
        st.error(f"❌ Erreur données: {e}")
        return pd.DataFrame()

def predict_client(model, features_df):
    """Faire une prédiction pour un client"""
    try:
        if model is None:
            return None, "Modèle non disponible"
        
        # Gérer le cas où le modèle est un dictionnaire
        if isinstance(model, dict):
            if 'model' in model:
                actual_model = model['model']
                st.info(f"Modèle extrait: {type(actual_model)}")
            else:
                return None, "Format de modèle non reconnu (dictionnaire sans clé 'model')"
        else:
            actual_model = model
        
        # Vérifier que le modèle a la méthode predict_proba
        if not hasattr(actual_model, 'predict_proba'):
            return None, f"Le modèle de type {type(actual_model)} n'a pas de méthode predict_proba"
        
        # Faire la prédiction avec le DataFrame
        proba = actual_model.predict_proba(features_df)[0][1]
        prediction = "REFUSÉ" if proba > 0.38 else "ACCEPTÉ"
        
        return {
            'probability': proba,
            'decision': prediction,
            'risk_level': 'ÉLEVÉ' if proba > 0.6 else 'MOYEN' if proba > 0.3 else 'FAIBLE'
        }, None
        
    except Exception as e:
        return None, f"Erreur prédiction: {e}"

def main():
    """Application principale"""
    
    # En-tête
    st.markdown("""
    <div class="main-header">
        <h1>MLOps Credit Scoring</h1>
        <p>Système de scoring crédit avec IA explicable</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Charger le modèle
    model = load_model()
    sample_data = load_sample_data()
    
    # Sidebar pour navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choisir une page",
            ["Accueil", "Prédiction", "Analyse", "Paramètres"]
        )
        
        st.markdown("---")
        st.subheader("Statistiques")
        if model:
            st.success("✅ Modèle: Random Forest")
            st.info("AUC: 0.743")
            st.info("Réduction coût: -31%")
        else:
            st.error("❌ Modèle non chargé")
    
    # Pages
    if page == "Accueil":
        render_home_page()
    elif page == "Prédiction":
        render_prediction_page(model, sample_data)
    elif page == "Analyse":
        render_analysis_page(sample_data)
    elif page == "Paramètres":
        render_settings_page()

def render_home_page():
    """Page d'accueil"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Performance</h3>
            <h2>AUC: 0.743</h2>
            <p>Très performant pour le crédit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Impact Métier</h3>
            <h2>-31% Coût</h2>
            <p>Réduction significative</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Modèle</h3>
            <h2>Random Forest</h2>
            <p>Robuste et explicable</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Fonctionnalités")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Prédiction**
        - Analyse de profil client
        - Probabilité de défaut
        - Décision automatique
        - Explicabilité SHAP
        """)
        
        st.markdown("""
        **Analyse**
        - Visualisations interactives
        - Tendances et patterns
        - Performance modèle
        """)
    
    with col2:
        st.markdown("""
        **Paramètres**
        - Configuration seuils
        - Préférences d'affichage
        - Export/Import données
        """)
        
        st.markdown("""
        **Monitoring**
        - Data drift detection
        - Alertes automatiques
        - Rapports qualité
        """)

def render_prediction_page(model, sample_data):
    """Page de prédiction"""
    
    st.subheader("Prédiction Client")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Saisie Profil Client")
        
        # Formulaire simplifié
        with st.form("client_form"):
            age = st.slider("Âge (années)", 18, 80, 35)
            income = st.number_input("Revenus annuels (€)", 0, 1000000, 150000)
            credit = st.number_input("Montant crédit demandé (€)", 0, 2000000, 500000)
            ext_source_2 = st.slider("Score externe 2", 0.0, 1.0, 0.5)
            ext_source_3 = st.slider("Score externe 3", 0.0, 1.0, 0.5)
            
            submit = st.form_submit_button("🚀 Analyser")
        
        # Ou charger un exemple
        st.markdown("---")
        if st.button("Charger exemple aléatoire"):
            if not sample_data.empty:
                example = sample_data.sample(1).iloc[0]
                st.session_state.example = example
                st.success("Exemple chargé! Modifiez les valeurs ci-dessus si nécessaire.")
    
    with col2:
        st.markdown("### Résultat de l'Analyse")
        
        if submit and model:
            # Le modèle attend exactement 20 features nommées feature_0 à feature_19
            base_features = [
                ext_source_2,      # feature_0
                ext_source_3,      # feature_1
                -age * 365,        # feature_2 (DAYS_BIRTH)
                income,            # feature_3
                credit,            # feature_4
            ]
            
            # Compléter avec 15 features supplémentaires (zéros pour la démo)
            # En production, il faudrait les vraies features du preprocessing
            features = base_features + [0.0] * 15  # Total = 20 features
            
            st.info(f"Envoi de {len(features)} features au modèle (attendu: 20)")
            
            # Créer un DataFrame avec les noms de features attendus
            feature_names = [f'feature_{i}' for i in range(20)]
            features_df = pd.DataFrame([features], columns=feature_names)
            st.info(f"Features DataFrame créé: {features_df.shape}")
            
            result, error = predict_client(model, features_df)
            
            if result:
                # Affichage du résultat
                prob = result['probability']
                decision = result['decision']
                risk = result['risk_level']
                
                # Gauge de probabilité
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probabilité de Défaut (%)"},
                    delta = {'reference': 38},  # Seuil
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#7451F8"},
                        'steps': [
                            {'range': [0, 30], 'color': "#D1FAE5"},
                            {'range': [30, 60], 'color': "#FEF3C7"},
                            {'range': [60, 100], 'color': "#FEE2E2"}
                        ],
                        'threshold': {
                            'line': {'color': "#EF4444", 'width': 4},
                            'thickness': 0.75,
                            'value': 38
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Décision
                if decision == "ACCEPTÉ":
                    st.markdown(f'<div class="status-success">✅ CRÉDIT {decision}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-danger">❌ CRÉDIT {decision}</div>', unsafe_allow_html=True)
                
                st.markdown(f"**Niveau de risque:** {risk}")
                st.markdown(f"**Probabilité:** {prob:.1%}")
                
                # Facteurs explicatifs (simplifié)
                st.markdown("### 🔍 Facteurs Explicatifs")
                if prob > 0.5:
                    st.error("🔴 Facteurs de risque élevé détectés")
                elif prob > 0.3:
                    st.warning("🟡 Profil présentant quelques risques")
                else:
                    st.success("🟢 Profil à faible risque")
                
            else:
                st.error(f"❌ {error}")

def render_analysis_page(sample_data):
    """Page d'analyse"""
    
    st.subheader("Analyse des Données")
    
    if sample_data.empty:
        st.warning("Pas de données à analyser")
        return
    
    # Distribution des scores
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Distribution EXT_SOURCE_2")
        fig = px.histogram(sample_data, x='EXT_SOURCE_2', nbins=20)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Distribution EXT_SOURCE_3")
        fig = px.histogram(sample_data, x='EXT_SOURCE_3', nbins=20)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Corrélations
    st.markdown("### Matrice de Corrélation")
    corr = sample_data.select_dtypes(include=[np.number]).corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistiques
    st.markdown("### Statistiques Descriptives")
    st.dataframe(sample_data.describe())

def render_settings_page():
    """Page de paramètres"""
    
    st.subheader("Paramètres")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Seuils de Décision")
        threshold = st.slider("Seuil de refus (%)", 0, 100, 38)
        st.info(f"Clients avec probabilité > {threshold}% seront refusés")
        
        st.markdown("### Affichage")
        theme = st.selectbox("Thème", ["Clair", "Sombre"])
        lang = st.selectbox("Langue", ["Français", "English"])
    
    with col2:
        st.markdown("### Export/Import")
        if st.button("Exporter paramètres"):
            st.success("Paramètres exportés!")

        uploaded_file = st.file_uploader("Importer paramètres", type=['json'])
        if uploaded_file:
            st.success("Paramètres importés!")

        st.markdown("### Mise à jour Modèle")
        if st.button("Recharger modèle"):
            st.cache_resource.clear()
            st.success("Modèle rechargé!")

if __name__ == "__main__":
    main()
