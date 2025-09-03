# Architecture du Projet MLOps Scoring Crédit

## Schéma d'Architecture Simplifié

```mermaid
graph TB
    %% Interface Utilisateur
    USER[Utilisateur]
    STREAMLIT[Application Streamlit<br/>Interface Web Locale]
    API[API FastAPI<br/>Production Cloud]

    %% Modèle ML
    MODEL[Modèle RandomForest<br/>153 Features<br/>AUC 0.736]

    %% Données
    DATA[Données Home Credit<br/>Features Client]

    %% Flux principal - Architecture Hybride
    USER --> STREAMLIT
    USER --> API
    STREAMLIT --> API
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
        STREAMLIT[Application Streamlit<br/>Interface Web Hybride<br/>Port Configurable]
        API_CLIENT[Client API<br/>Applications externes]
    end

    subgraph "Services Backend"
        API[API FastAPI<br/>Service de Prédiction<br/>Port 8000]
        SECURITY[Module Sécurité<br/>Authentification<br/>Rate Limiting]
        LOGS[Système de Logs<br/>JSON Structuré]
    end

    subgraph "Pipeline Machine Learning"
        PREP[Preprocessing<br/>Encodage Features<br/>Validation Données]
        MODEL[Modèle RandomForest<br/>153 Features Engineered<br/>Score Métier AUC 0.736]
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

    %% Flux principal - Architecture Hybride
    USER --> STREAMLIT
    USER --> API_CLIENT
    API_CLIENT --> API
    STREAMLIT --> API
    STREAMLIT --> MODEL

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

## Architecture Hybride - Innovation Clé

```mermaid
graph TB
    subgraph "Mode de Fonctionnement Intelligent"
        USER[Utilisateur]
        STREAMLIT[Interface Streamlit]

        subgraph "Détection Automatique"
            ENV_CHECK{Environment<br/>Production ?}
            API_CHECK{API Distante<br/>Disponible ?}
        end

        subgraph "Mode Production"
            API_REMOTE[API Render.com<br/>https://mn-opc-7025.onrender.com]
            MODEL_REMOTE[Modèle Cloud<br/>Scalable]
        end

        subgraph "Mode Local (Fallback)"
            MODEL_LOCAL[Modèle Local<br/>models/best_credit_model.pkl]
            CACHE_LOCAL[Cache Local<br/>Continuité Service]
        end
    end

    %% Flux intelligent
    USER --> STREAMLIT
    STREAMLIT --> ENV_CHECK
    ENV_CHECK -->|Production| API_CHECK
    ENV_CHECK -->|Développement| MODEL_LOCAL
    API_CHECK -->|Disponible| API_REMOTE
    API_CHECK -->|Indisponible| MODEL_LOCAL
    API_REMOTE --> MODEL_REMOTE
    MODEL_LOCAL --> CACHE_LOCAL

    %% Styles
    classDef production fill:#90EE90
    classDef fallback fill:#FFB6C1
    classDef decision fill:#87CEEB

    class API_REMOTE,MODEL_REMOTE production
    class MODEL_LOCAL,CACHE_LOCAL fallback
    class ENV_CHECK,API_CHECK decision
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
        B2[Endpoints REST<br/>/predict, /predict_public, /batch_predict<br/>/health, /explain, /feature_importance]
        B3[Middleware<br/>CORS, Security, Rate Limiting]
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

## Flux de Données Détaillé - Architecture Hybride

```mermaid
sequenceDiagram
    participant U as Utilisateur
    participant S as Streamlit
    participant ENV as Détection ENV
    participant API as API Distante
    participant ML as Modèle Local
    participant R as Résultat

    Note over S,ENV: Architecture Hybride Intelligente

    %% Flux principal avec détection
    U->>S: Saisie données client
    S->>S: Validation côté client
    S->>ENV: Vérifier environnement

    alt Mode Production
        ENV->>API: Tenter connexion API
        alt API Disponible
            S->>API: POST /predict_public
            API->>API: Preprocessing + Prédiction
            API->>R: Résultat JSON
            R->>S: Affichage résultat
        else API Indisponible
            Note over S,ML: Fallback Automatique
            S->>ML: Prédiction locale
            ML->>R: Résultat local
            R->>S: Affichage avec warning
        end
    else Mode Développement
        S->>ML: Prédiction locale directe
        ML->>R: Résultat local
        R->>S: Affichage résultat
    end

    S->>U: Interface utilisateur finale

    %% Flux API direct (externe)
    Note over U,API: Usage API Direct
    U->>API: POST /predict_public (direct)
    API->>API: Authentification + Rate Limiting
    API->>API: Validation + Preprocessing
    API->>API: Prédiction modèle
    API->>U: Réponse JSON structurée
```

## Composants Techniques

| Couche         | Composant      | Technologie           | Rôle                          |
| -------------- | -------------- | --------------------- | ----------------------------- |
| **Frontend**   | Streamlit App  | Python + Streamlit    | Interface hybride utilisateur |
| **API**        | FastAPI Server | Python + FastAPI      | Service REST cloud            |
| **ML**         | Model Engine   | Scikit-learn RF       | Prédictions (153 features)    |
| **Data**       | Feature Store  | Pandas + Joblib       | Engineering + Gestion         |
| **Monitoring** | Analytics      | SHAP + Evidently 0.7+ | Surveillance + Drift          |
| **Security**   | Auth Layer     | JWT + Rate Limiting   | Protection API                |

## Points d'Intégration Clés

### 1. **Architecture Hybride Innovante**

- **Innovation** : Détection automatique environnement + fallback intelligent
- **Production** : API distante Render.com en priorité
- **Développement** : Modèle local pour tests rapides
- **Résilience** : Continuité service même si API indisponible
- **Configuration** : Variables d'environnement (RENDER, STREAMLIT_ENV)

### 2. **Modèle Unique Multi-Usage**

- **Fichier** : `models/best_credit_model.pkl`
- **Usage** : API + Streamlit (architecture hybride)
- **Features** : 153 variables après feature engineering
- **Performance** : AUC 0.736, coût métier optimisé à 7,100

### 3. **Preprocessing Cohérent**

- **API** : `api/app.py` - `preprocess_input()`
- **Streamlit** : `main.py` - `create_full_feature_set()`
- **Logique** : Même encodage + feature engineering partout
- **Validation** : 153 features identiques API ↔ Local

### 4. **Validation Stricte Multi-Niveau**

- **Pydantic** : Modèles de validation API (30+ champs)
- **Types** : Validation automatique + sanitisation
- **Business Rules** : Validation métier avant prédiction
- **Erreurs** : Gestion centralisée avec codes HTTP appropriés

### 5. **Monitoring Unifié Intelligent**

- **Logs** : JSON structuré avec rotation automatique
- **Métriques** : Temps réel (latence, throughput, erreurs)
- **Health Checks** : Endpoints de surveillance automatique
- **Alertes** : Détection proactive des anomalies

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
