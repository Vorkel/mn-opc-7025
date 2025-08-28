# Guide Utilisateur - API de Scoring Crédit

## Vue d'ensemble

L'API de Scoring Crédit est un système intelligent qui évalue automatiquement la probabilité qu'un client rembourse son crédit. Elle utilise des algorithmes de Machine Learning avancés pour analyser le profil financier du client et fournir une décision d'octroi de crédit.

### URL de l'API

**Production** : `https://mn-opc-7025.onrender.com`

---

### Fonctionnalités principales

### Scoring individuel
- Évaluation instantanée d'un client
- Probabilité de défaut précise
- Décision automatique (ACCORDÉ/REFUSÉ)
- Niveau de risque (FAIBLE/MOYEN/ÉLEVÉ)

### Scoring en lot
- Traitement multiple de clients
- Optimisation pour volumes importants
- Rapport de synthèse automatique

### Explicabilité
- Analyse SHAP des facteurs influents
- Explication des décisions
- Transparence réglementaire

### Monitoring
- Dashboard temps réel
- Métriques de performance
- Alertes automatiques

---

## Utilisation de l'API

### 1. Endpoint principal : Scoring individuel

```http
POST /predict
Content-Type: application/json
```

**Exemple de requête** :
```json
{
  "CODE_GENDER": "M",
  "FLAG_OWN_CAR": "Y",
  "FLAG_OWN_REALTY": "Y",
  "CNT_CHILDREN": 1,
  "AMT_INCOME_TOTAL": 150000,
  "AMT_CREDIT": 300000,
  "AMT_ANNUITY": 15000,
  "AMT_GOODS_PRICE": 280000,
  "NAME_INCOME_TYPE": "Working",
  "NAME_EDUCATION_TYPE": "Higher education",
  "NAME_FAMILY_STATUS": "Married",
  "NAME_HOUSING_TYPE": "House / apartment",
  "DAYS_BIRTH": -12000,
  "DAYS_EMPLOYED": -3000,
  "DAYS_REGISTRATION": 0,
  "DAYS_ID_PUBLISH": -1000,
  "FLAG_MOBIL": 1,
  "FLAG_EMP_PHONE": 1,
  "FLAG_WORK_PHONE": 0,
  "FLAG_CONT_MOBILE": 1
}
```

**Réponse** :
```json
{
  "client_id": "CLIENT_001",
  "probability": 0.15,
  "decision": "ACCORDÉ",
  "threshold": 0.48,
  "risk_level": "FAIBLE",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 2. **Endpoint : Scoring en lot**

```http
POST /batch_predict
Content-Type: application/json
```

**Exemple** :
```json
[
  {
    "CODE_GENDER": "M",
    "AMT_INCOME_TOTAL": 150000,
    ...
  },
  {
    "CODE_GENDER": "F",
    "AMT_INCOME_TOTAL": 120000,
    ...
  }
]
```

### 3. **Endpoints utilitaires**

#### Santé de l'API
```http
GET /health
```

#### Informations du modèle
```http
GET /model/info
```

#### Documentation interactive
```http
GET /docs
```

---

## Champs obligatoires

### Informations personnelles
- `CODE_GENDER` : Genre (M/F)
- `FLAG_OWN_CAR` : Possède une voiture (Y/N)
- `FLAG_OWN_REALTY` : Possède un bien immobilier (Y/N)
- `CNT_CHILDREN` : Nombre d'enfants (entier ≥ 0)

### Informations financières
- `AMT_INCOME_TOTAL` : Revenu total annuel (€)
- `AMT_CREDIT` : Montant du crédit demandé (€)
- `AMT_ANNUITY` : Montant de l'annuité (€)
- `AMT_GOODS_PRICE` : Prix des biens financés (€)

### Informations professionnelles
- `NAME_INCOME_TYPE` : Type de revenu
  - "Working" (Salarié)
  - "Commercial associate" (Associé commercial)
  - "Pensioner" (Retraité)
  - "State servant" (Fonctionnaire)
  - "Unemployed" (Sans emploi)

### Informations sociales
- `NAME_EDUCATION_TYPE` : Niveau d'éducation
  - "Higher education" (Études supérieures)
  - "Secondary / secondary special" (Secondaire)
  - "Incomplete higher" (Études supérieures incomplètes)
  - "Lower secondary" (Collège)

- `NAME_FAMILY_STATUS` : Statut familial
  - "Married" (Marié)
  - "Single / not married" (Célibataire)
  - "Civil marriage" (Union libre)
  - "Widow" (Veuf/Veuve)
  - "Separated" (Séparé)

### Informations logement
- `NAME_HOUSING_TYPE` : Type de logement
  - "House / apartment" (Maison/Appartement)
  - "With parents" (Chez les parents)
  - "Municipal apartment" (Logement social)
  - "Rented apartment" (Location)
  - "Office apartment" (Logement de fonction)

### Informations temporelles
- `DAYS_BIRTH` : Âge en jours (négatif, ex: -12000)
- `DAYS_EMPLOYED` : Jours d'emploi (négatif si employé)
- `DAYS_REGISTRATION` : Jours depuis l'enregistrement
- `DAYS_ID_PUBLISH` : Jours depuis émission pièce d'identité

### Informations contact
- `FLAG_MOBIL` : Possède un mobile (0/1)
- `FLAG_EMP_PHONE` : Téléphone professionnel (0/1)
- `FLAG_WORK_PHONE` : Téléphone au travail (0/1)
- `FLAG_CONT_MOBILE` : Contact mobile (0/1)

---

## Interprétation des résultats

### Probabilité de défaut
- **0.0 - 0.2** : 🟢 **Risque FAIBLE** - Client très fiable
- **0.2 - 0.5** : 🟡 **Risque MOYEN** - Évaluation case par case
- **0.5 - 1.0** : 🔴 **Risque ÉLEVÉ** - Client à risque

### Seuil de décision
Le seuil optimal est calculé pour **minimiser les coûts métier** :
- **Faux Négatif** (mauvais client accepté) = **10x** plus coûteux
- **Faux Positif** (bon client refusé) = **1x** coût de base

### Niveaux de risque
- **FAIBLE** : Probabilité < 20% - Accord recommandé
- **MOYEN** : Probabilité 20-50% - Analyse complémentaire
- **ÉLEVÉ** : Probabilité > 50% - Refus recommandé

---

## Interface Streamlit

### Accès
L'interface utilisateur graphique est disponible via Streamlit pour une utilisation conviviale sans programmation.

### Fonctionnalités
- **Saisie guidée** des informations client
- **Validation automatique** des données
- **Résultats visuels** avec graphiques
- **Historique** des évaluations
- **Export PDF** des rapports

### Scoring en lot
- Upload de fichiers CSV
- Traitement automatique
- Rapport de synthèse
- Export des résultats

---

## Dashboard de monitoring

### URL du monitoring
Accès au dashboard de surveillance : `/monitoring/dashboard.py`

### Métriques surveillées
- **Disponibilité** de l'API
- **Temps de réponse** moyen
- **Taux de succès** des requêtes
- **Volume** de prédictions
- **Alertes automatiques**

### Alertes configurées
- Temps de réponse > 1 seconde
- Taux d'erreur > 5%
- API inaccessible
- Charge système élevée

---

## ❓ FAQ - Questions fréquentes

### **Q: Combien de temps prend une prédiction ?**
**R:** Moins de 500ms en moyenne pour une prédiction individuelle.

### **Q: Puis-je traiter plusieurs clients en même temps ?**
**R:** Oui, utilisez l'endpoint `/batch_predict` pour des volumes importants.

### **Q: Comment interpréter la probabilité de défaut ?**
**R:** C'est la probabilité (0-100%) que le client ne rembourse pas son crédit.

### **Q: Le seuil de décision est-il fixe ?**
**R:** Non, il est optimisé selon les coûts métier (FN=10x, FP=1x).

### **Q: Puis-je obtenir des explications sur la décision ?**
**R:** Oui, l'API fournit l'analyse SHAP des facteurs influents.

### **Q: Les données sont-elles sécurisées ?**
**R:** Oui, HTTPS obligatoire et logs structurés pour audit.

### **Q: Que faire si l'API est lente ?**
**R:** Consultez le dashboard de monitoring pour identifier les goulots.

### **Q: Comment signaler un problème ?**
**R:** Les logs automatiques permettent un diagnostic rapide.

---

## Glossaire métier

### Scoring crédit
Technique d'évaluation automatique de la solvabilité d'un emprunteur utilisant des algorithmes statistiques.

### Probabilité de défaut
Estimation quantitative du risque qu'un client ne rembourse pas son crédit dans les délais convenus.

### Seuil de décision
Valeur critique de probabilité au-dessus de laquelle le crédit est refusé. Optimisé selon les coûts métier.

### Faux Négatif (FN)
Erreur où un "mauvais" client est prédit comme "bon" → crédit accordé à risque → perte financière.

### Faux Positif (FP)
Erreur où un "bon" client est prédit comme "mauvais" → crédit refusé → manque à gagner.

### Machine Learning
Techniques d'intelligence artificielle permettant aux algorithmes d'apprendre à partir de données historiques.

### SHAP (SHapley Additive exPlanations)
Méthode d'explicabilité qui quantifie la contribution de chaque caractéristique client à la décision finale.

### MLOps
Pratiques d'industrialisation des modèles de Machine Learning (déploiement, monitoring, mise à jour).

### Déséquilibre de classes
Situation où il y a beaucoup plus de "bons" clients que de "mauvais" dans les données d'entraînement.

### Data Drift
Phénomène où les caractéristiques des nouvelles données diffèrent de celles utilisées pour l'entraînement.

### AUC-ROC
Métrique technique mesurant la capacité du modèle à distinguer les bons des mauvais clients.

### Score métier
Métrique personnalisée intégrant les coûts réels des erreurs de prédiction (FN plus coûteux que FP).

---

## Support technique

### Contact
- Documentation : `/docs` (Swagger UI)
- Monitoring : Dashboard temps réel disponible
- Logs : Système de logging JSON structuré

### Debugging
1. Vérifiez le statut via `/health`
2. Consultez les logs d'erreur
3. Testez avec des données d'exemple
4. Vérifiez le format des requêtes JSON

### Optimisation
- Utilisez `/batch_predict` pour les volumes
- Mise en cache côté client si nécessaire
- Monitoring proactif des performances

---

## Sécurité et conformité

### Sécurité
- HTTPS obligatoire en production
- Validation stricte des entrées
- Logs d'audit complets
- Pas de stockage des données sensibles

### Conformité RGPD
- Pas de données personnelles stockées
- Anonymisation possible
- Droit à l'explication respecté via SHAP
- Traçabilité des décisions

### Réglementation bancaire
- Explicabilité des décisions (SHAP)
- Audit trail complet
- Monitoring continu
- Documentation métier complète

---
