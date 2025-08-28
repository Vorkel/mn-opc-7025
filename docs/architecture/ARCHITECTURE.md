# Architecture du Projet MLOps Scoring Crédit

## Schéma d'Architecture Simplifié

```mermaid
graph TB
    %% Interface Utilisateur
    USER[Utilisateur]
    STREAMLIT[Application Streamlit<br/>Interface Web Locale]
    API[API FastAPI<br/>Production Cloud]

    %% Modèle ML
    MODEL[Modèle RandomForest<br/>Scoring Crédit]

    %% Données
    DATA[Données Home Credit<br/>Features Client]

    %% Flux principal
    USER --> STREAMLIT
    USER --> API
    STREAMLIT --> MODEL
    API --> MODEL
    DATA --> MODEL

    %% Résultats
    MODEL --> RESULT[Résultat Prédiction<br/>Probabilité de défaut<br/>Décision Accordé/Refusé<br/>Niveau de risque]

    RESULT --> STREAMLIT
    RESULT --> API

    %% Styles
    classDef user fill:#ff9999
    classDef app fill:#99ccff
    classDef model fill:#99ff99
    classDef data fill:#ffcc99
    classDef result fill:#cc99ff

    class USER user
    class STREAMLIT,API app
    class MODEL model
    class DATA data
    class RESULT result
```

## Schéma d'Architecture Détaillé

```mermaid
graph TB
    %% Utilisateur et Interfaces
    USER[Utilisateur Final]

    subgraph "Interfaces Utilisateur"
        STREAMLIT[Application Streamlit<br/>Interface Web Locale<br/>Port 8501]
        API_CLIENT[Client API<br/>Applications externes]
    end

    subgraph "Services Backend"
        API[API FastAPI<br/>Service de Prédiction<br/>Port 8000]
        SECURITY[Module Sécurité<br/>Authentification<br/>Rate Limiting]
        LOGS[Système de Logs<br/>JSON Structuré]
    end

    subgraph "Pipeline Machine Learning"
        PREP[Preprocessing<br/>Encodage Features<br/>Validation Données]
        MODEL[Modèle RandomForest<br/>16 Features Optimisées<br/>Score Métier]
        EVAL[Évaluation<br/>Métriques Performance<br/>Seuil Optimisé]
    end

    subgraph "Gestion des Données"
        RAW[Données Brutes<br/>Home Credit Dataset]
        PROC[Données Traitées<br/>Features Engineering<br/>Encodage]
        CACHE[Cache Modèle<br/>Chargement Lazy]
    end

    subgraph "Monitoring & Analytics"
        MONITOR[Dashboard Monitoring<br/>Métriques Temps Réel]
        DRIFT[Data Drift Detection<br/>Surveillance Modèle]
        REPORTS[Rapports Analytics<br/>Visualisations SHAP]
    end

    %% Flux principal
    USER --> STREAMLIT
    USER --> API_CLIENT
    API_CLIENT --> API
    STREAMLIT --> API

    %% Flux API
    API --> SECURITY
    API --> PREP
    PREP --> MODEL
    MODEL --> EVAL
    EVAL --> API

    %% Flux données
    RAW --> PROC
    PROC --> PREP
    CACHE --> MODEL

    %% Flux monitoring
    API --> LOGS
    MODEL --> DRIFT
    EVAL --> MONITOR
    DRIFT --> REPORTS

    %% Résultats
    API --> RESULT[Résultat Final<br/>Probabilité de Défaut<br/>Décision Métier<br/>Niveau de Risque]
    RESULT --> STREAMLIT
    RESULT --> API_CLIENT

    %% Styles
    classDef user fill:#ff9999
    classDef interface fill:#99ccff
    classDef service fill:#99ff99
    classDef ml fill:#ffcc99
    classDef data fill:#cc99ff
    classDef monitor fill:#ff99cc
    classDef result fill:#99ffcc

    class USER user
    class STREAMLIT,API_CLIENT interface
    class API,SECURITY,LOGS service
    class PREP,MODEL,EVAL ml
    class RAW,PROC,CACHE data
    class MONITOR,DRIFT,REPORTS monitor
    class RESULT result
```

## Architecture Technique Détaillée

```mermaid
graph LR
    subgraph "Frontend Layer"
        A1[Streamlit App<br/>Interface Web]
        A2[Formulaires<br/>Validation Client]
        A3[Visualisations<br/>Graphiques Plotly]
        A4[Historique<br/>Session State]
    end

    subgraph "API Layer"
        B1[FastAPI Server<br/>Uvicorn]
        B2[Endpoints REST<br/>/predict, /batch_predict]
        B3[Middleware<br/>CORS, Security]
        B4[Validation<br/>Pydantic Models]
    end

    subgraph "ML Layer"
        C1[Model Loader<br/>Joblib]
        C2[Preprocessing<br/>Feature Engineering]
        C3[Prediction Engine<br/>RandomForest]
        C4[Business Logic<br/>Seuil Métier]
    end

    subgraph "Data Layer"
        D1[Raw Data<br/>CSV Files]
        D2[Processed Data<br/>Encoded Features]
        D3[Model Files<br/>.pkl Files]
        D4[Cache System<br/>Memory/Redis]
    end

    subgraph "Analytics Layer"
        E1[Feature Importance<br/>SHAP Analysis]
        E2[Performance Metrics<br/>AUC, Accuracy]
        E3[Data Drift<br/>Evidently 0.7+]
        E4[Reports<br/>PDF, CSV]
    end

    %% Connexions Frontend-API
    A1 --> B1
    A2 --> B2
    A3 --> B4
    A4 --> B3

    %% Connexions API-ML
    B1 --> C1
    B2 --> C2
    B3 --> C3
    B4 --> C4

    %% Connexions ML-Data
    C1 --> D3
    C2 --> D2
    C3 --> D4
    C4 --> D1

    %% Connexions Analytics
    C3 --> E1
    C4 --> E2
    D2 --> E3
    E1 --> E4
```

## Flux de Données Détaillé

```mermaid
sequenceDiagram
    participant U as Utilisateur
    participant S as Streamlit
    participant A as API
    participant P as Preprocessing
    participant M as Modèle
    participant D as Données
    participant R as Résultat

    %% Flux Streamlit
    U->>S: Saisie données client
    S->>S: Validation côté client
    S->>A: POST /predict
    A->>P: Preprocessing données
    P->>M: Prédiction modèle
    M->>R: Résultat (probabilité)
    R->>A: JSON response
    A->>S: Affichage résultat
    S->>U: Interface utilisateur

    %% Flux API direct
    U->>A: POST /predict (direct)
    A->>P: Validation + preprocessing
    P->>M: Prédiction
    M->>R: Résultat métier
    R->>A: Réponse API
    A->>U: JSON final

    %% Flux batch
    U->>A: POST /batch_predict
    A->>P: Preprocessing lot
    P->>M: Prédictions multiples
    M->>R: Résultats batch
    R->>A: Rapport synthèse
    A->>U: CSV résultats
```

## Composants Techniques

| Couche         | Composant      | Technologie           | Rôle                  |
| -------------- | -------------- | --------------------- | --------------------- |
| **Frontend**   | Streamlit App  | Python + Streamlit    | Interface utilisateur |
| **API**        | FastAPI Server | Python + FastAPI      | Service REST          |
| **ML**         | Model Engine   | Scikit-learn          | Prédictions           |
| **Data**       | Feature Store  | Pandas + Joblib       | Gestion données       |
| **Monitoring** | Analytics      | SHAP + Evidently 0.7+ | Surveillance          |

## Points d'Intégration Clés

### 1. **Modèle Unique**

- **Fichier** : `models/best_credit_model.pkl`
- **Usage** : API + Streamlit
- **Features** : 16 variables standardisées

### 2. **Preprocessing Cohérent**

- **API** : `api/app.py` - `preprocess_input()`
- **Streamlit** : `main.py` - `preprocess_for_prediction()`
- **Logique** : Même encodage partout

### 3. **Validation Stricte**

- **Pydantic** : Modèles de validation
- **Types** : Validation automatique
- **Erreurs** : Gestion centralisée

### 4. **Monitoring Unifié**

- **Logs** : JSON structuré
- **Métriques** : Temps réel
- **Alertes** : Proactives

## Sécurité et Performance

### **Sécurité**

- **HTTPS** : Obligatoire en production
- **Rate Limiting** : Protection contre le spam
- **Validation** : Données strictement contrôlées
- **Logs** : Audit trail complet

### **Performance**

- **Cache** : Modèle en mémoire
- **Lazy Loading** : Chargement à la demande
- **Compression** : Réponses optimisées
- **Monitoring** : Métriques temps réel
