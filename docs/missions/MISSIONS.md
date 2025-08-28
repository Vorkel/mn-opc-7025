# Cahier des charges unifié: modélisation, MLOps, déploiement API et monitoring du drift.

## Cahier des charges — Implémentez un modèle de scoring (missions 1 et 2)

### 1. Contexte et objectifs
 • Contexte: Société financière “Prêt à dépenser” souhaitant évaluer le risque de défaut pour des clients avec peu/pas d’historique. Deux missions successives: 1) élaboration du modèle et socle MLOps, 2) déploiement, interface de test et monitoring du data drift.
 • Objectif général: Construire, expliquer et déployer un modèle de classification prédisant la probabilité de défaut, décider l’acceptation/refus via un seuil optimisé métier, et mettre en place un cycle MLOps complet incluant suivi du drift.

### 2. Périmètre fonctionnel
 • Données multi-tables: Jointure des tables fournies (liens “les données”). Jeux “application_train” (modélisation) et “application_test” (simulation données prod) pour l’analyse de drift.
 • Modèle: Classification binaire avec gestion du déséquilibre de classes. Sortie: probabilité de défaut et classe décisionnelle via seuil métier optimisé.
 • Explicabilité: Importance des features globale et locale pour comprendre et justifier le score.
 • Exposition: API de scoring déployée sur un cloud (préférence gratuite) + interface de test locale (Notebook ou Streamlit).
 • MLOps: MLflow pour tracking/registry/serving, Git/GitHub, GitHub Actions (CI/CD), tests automatisés, monitoring data drift avec Evidently.

### 3. Exigences détaillées — Mission 1: Élaborez le modèle de scoring

#### 3.1 Préparation de l’environnement d’expérimentation
 • MLflow Tracking: journaliser hyperparamètres, métriques, artefacts (modèles, figures, rapports).
 • MLflow UI: comparer les runs et visualiser les expériences.
 • Model Registry MLflow: versionner les modèles (Staging/Production), gérer la promotion/rollback.
 • MLflow Serving (test): valider le serving local pour un modèle enregistré.

#### 3.2 Données et préparation
 • Kernels Kaggle (optionnel): s’inspirer des kernels fournis pour EDA et feature engineering, en les adaptant au contexte et aux contraintes du projet.
 • Préparation: nettoyage, gestion des valeurs manquantes, encodage, normalisation, dérivation de variables, jointures cohérentes, séparation stricte train/valid/test, versionnage des scripts/datasets dérivés.

#### 3.3 Score métier et stratégie d’évaluation
 • Coût différencié: le coût d’un FN est 10× celui d’un FP (hypothèse métier).
 • Fonction de coût: définir un score métier pondéré FN/FP et l’utiliser pour comparer les modèles.
 • Seuil décisionnel: optimiser le seuil sur validation pour minimiser le coût métier (ne pas se contenter de 0,5).
 • Métriques de contrôle: suivre AUC et accuracy en parallèle; si AUC > 0,82, analyser le risque d’overfitting.

#### 3.4 Modélisation et sélection
 • Baselines: établir une baseline (ex: régression logistique, arbre simple) pour référence.
 • Algorithmes: tester plusieurs classifieurs (ex: arbres/boosting, linéaires, etc.) avec gestion du déséquilibre (class_weight, rééchantillonnage).
 • Validation: Cross-Validation et GridSearchCV (ou équivalent) pour l’optimisation hyperparamètres.
 • Sélection finale: choisir sur coût métier (avec seuil optimisé), vérifier robustesse via AUC/accuracy sur un set de test tenu à l’écart.

#### 3.5 Explicabilité
 • Globale: importance des variables (selon le modèle, ou via méthodes globales).
 • Locale: explications par individu (score client) pour justifier une décision.

### 4. Exigences détaillées — Mission 2: Intégrez et optimisez le système MLOps

#### 4.1 Déploiement API dans le cloud
 • Versionnage: gestion du code avec Git, dépôt GitHub.
 • CI/CD: GitHub Actions pour build, tests et déploiement automatisé de l’API.
 • Hébergement: déploiement sur une solution cloud (préférence gratuite), configuration externe (seuil métier, version modèle, endpoints MLflow si applicable).

#### 4.2 Interface de test
 • Streamlit (ou Notebook): formulaire d’entrée des features, appel de l’API, affichage probabilité, classe (avec seuil métier), et explications locales synthétiques.

#### 4.3 Monitoring Data Drift (Evidently)
 • Hypothèse: “application_train” = référence (modélisation), “application_test” = proxy données de production.
 • Rapport drift: générer un rapport HTML Evidently sur les principales features (distribution, statistiques, tests) pour détecter du drift potentiel.
 • Intégration: conserver le rapport en artefact et documenter la lecture/interprétation et les actions envisagées en cas de drift (alerte, re-entrainement, recalibrage).

### 5. Spécifications API
 • Entrée: schéma des features nécessaires (types, contraintes, gestion des valeurs manquantes côté API), validation stricte et messages d’erreur clairs.
 • Sortie:
 ▫ probabilité_de_defaut: float [0,1]
 ▫ classe: “accepte”/“refuse” basée sur le seuil optimisé métier
 ▫ meta: version du modèle (depuis registry), timestamp, latence
 ▫ explications_locales: top facteurs (optionnel si coût performance important)
 • Non-fonctionnel: latence raisonnable, logs structurés, traçabilité requêtes (échantillonnage pour drift), tolérance aux champs inconnus selon politique.

### 6. Architecture et MLOps
 • Tracking/Registry: MLflow pour runs et versions modèles.
 • CI/CD:
 ▫ pipeline lint + tests (Pytest/Unittest) + build
 ▫ déploiement conditionnel par branche/tag/version du modèle
 • Observabilité: logs API (taux d’erreurs, latence), artefacts de drift (rapports Evidently), capacité de rollback rapide (promotion/dé-promotion dans le Model Registry).
 • Sécurité & conformité: gestion des secrets (GitHub Secrets), validation des entrées, journalisation conforme, documentation d’usage.

### 7. Livrables
 • Code:
 ▫ scripts/notebooks de data prep, entraînement, évaluation, explicabilité
 ▫ service API (endpoints, schémas, validation)
 ▫ interface Streamlit/Notebook de test
 ▫ workflows GitHub Actions (CI/CD)
 ▫ scripts MLflow (tracking, registry, serving test)
 ▫ script/rapport Evidently
 • Artefacts:
 ▫ runs MLflow, modèles versionnés (Staging/Production)
 ▫ rapport EDA, rapport explicabilité (globale/locale), rapport optimisation de seuil
 ▫ rapport HTML Evidently (drift)
 • Docs:
 ▫ README (installation, exécution, structure)
 ▫ spécification API (payloads exemples)
 ▫ guide MLOps (tracking, registry, serving, CI/CD)
 ▫ note de décision (choix modèle, gestion déséquilibre, seuil métier)

### 8. Critères d’acceptation
 • Modèle:
 ▫ gestion du déséquilibre justifiée et testée
 ▫ seuil métier optimisé et documenté (FN=10×FP par défaut)
 ▫ rapport AUC/accuracy + alerte/contrôle si AUC > 0,82
 • MLOps:
 ▫ MLflow opérationnel (tracking, UI, registry, serving testé)
 ▫ dépôt GitHub avec CI/CD fonctionnel, tests unitaires passants
 • API & Interface:
 ▫ API déployée sur cloud, stable, versionnée
 ▫ interface Streamlit/Notebook interrogeant l’API et affichant proba/classe/explications
 • Monitoring:
 ▫ rapport Evidently produit et interprété (train vs test)
 ▫ plan d’action en cas de drift (au moins décrit)
 • Documentation:
 ▫ complète, reproductible, et cohérente avec les livrables

### 9. Plan de mise en œuvre (ordre chronologique)
 1. Environnement & MLOps socle
 ▫ MLflow (tracking + UI + registry), dépôt GitHub initial, CI de base, structure de projet.
 2. Données & EDA
 ▫ jointures, nettoyage, features, adaptation des kernels (si utilisés), versionnage des datasets dérivés.
 3. Baselines & coût métier
 ▫ baseline, définition fonction de coût, protocole d’optimisation de seuil.
 4. Itérations de modélisation
 ▫ gestion déséquilibre, CV + GridSearch, sélection sur coût métier + contrôle AUC/accuracy.
 5. Explicabilité
 ▫ importance globale, cas clients avec importance locale.
 6. Packaging modèle & registry
 ▫ enregistrement MLflow, test de serving, promotion en Staging.
 7. API
 ▫ endpoints, validation, intégration du modèle et du seuil, tests unitaires.
 8. CI/CD & déploiement
 ▫ workflows GitHub Actions, déploiement sur cloud, variables d’environnement.
 9. Interface de test
 ▫ Streamlit/Notebook, scénarios de test représentatifs.
 10. Monitoring drift
 ▫ script Evidently (train vs test), rapport HTML, interprétation et doc.
 11. Stabilisation & docs
 ▫ README, spéc API, guide MLOps, préparation soutenance.

### 10. Risques & mitigations
 • Overfitting (AUC anormalement élevé): validation stricte, jeux hold-out, régularisation, réduction de fuite de cible.
 • Déséquilibre de classes: tester class_weight, SMOTE/undersampling, ajustement de seuil, calibration.
 • Seuil non optimal en prod: revalidation périodique, recalcul sur données récentes.
 • Drift en prod: rapports Evidently planifiés, seuils d’alerte, plan de re-entrainement.
 • Déploiement instable: blue/green via registry (Staging/Prod), rollback rapide.
 • Dette technique: CI stricte, tests unitaires, linting, documentation vivante.

### 11. Indicateurs clés
 • Modélisation: coût métier (avec seuil optimisé), AUC, accuracy.
 • Exploitation: disponibilité API, latence p95, taux d’erreurs.
 • Monitoring: indicateurs de drift (parties de features en alerte), fréquence de dérive, temps de réaction.

Ce cahier des charges unifie les deux missions dans l’ordre, en assurant continuité entre modélisation, MLOps, déploiement, interface de test et monitoring du drift avec Evidently.