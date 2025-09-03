# Guide de Validation du Projet - Credit Scoring MLOps

## Vue d'ensemble

Ce guide permet de valider que tous les composants du projet fonctionnent correctement avant la soutenance.

## Tests à exécuter

### 1. Tests unitaires

```bash
poetry run pytest tests/unit/ -v
```

**Attendu** : Tous les tests passent (BusinessScorer, feature engineering, validation données)

### 2. Tests d'intégration

```bash
poetry run pytest tests/integration/ -v
```

**Attendu** : Pipeline complet, MLflow, API fonctionnels (certains peuvent être ignorés)

### 3. Tests API

```bash
poetry run pytest tests/api/ -v
```

**Attendu** : Tests d'import validés (tests de fonctionnalité peuvent être ignorés)

### 4. Tests de performance

```bash
poetry run pytest tests/performance/ -v
```

**Attendu** : Import API < 2s, mémoire < 100MB, calculs < 1s

### 5. Tests de sécurité

```bash
poetry run pytest tests/security/ -v
```

**Attendu** : Aucun secret en dur, imports sécurisés

## Validation des fonctionnalités

### MLflow Tracking

- [ ] **Interface accessible** : `./scripts/launch_mlflow.sh`
- [ ] **URL** : http://localhost:5000 (ou port suivant)
- [ ] **Runs visibles** : Au moins 1 expérimentation avec des runs
- [ ] **Métriques** : Business Score, AUC, paramètres trackés

### Evidently Data Drift

- [ ] **Migration 0.7+** : `python -c "import evidently; print(evidently.__version__)"`
- [ ] **Tests fonctionnels** : `python tests/performance/test_simple_performance.py`
- [ ] **Rapport HTML** : Généré automatiquement

### API FastAPI

- [ ] **Import fonctionnel** : `python -c "from api.app import app; print('OK')"`
- [ ] **Endpoints** : Health, predict, batch_predict disponibles
- [ ] **Sécurité** : Authentification, rate limiting configurés

### Interface Streamlit

- [ ] **Application** : `streamlit run streamlit_app/main.py`
- [ ] **Formulaire** : Champs de saisie fonctionnels
- [ ] **Prédictions** : Résultats affichés correctement

## Métriques de validation

### Tests

- **Tests unitaires** : 37 tests passés
- **Tests d'intégration** : 6 tests passés
- **Tests API** : 12 tests passés
- **Tests performance** : 4 tests passés
- **Tests sécurité** : 4 tests passés

### Couverture

- **Couverture globale** : 26% (améliorable)
- **Modules principaux** : src/, api/, streamlit_app/

### Performance

- **Import API** : < 2 secondes
- **Utilisation mémoire** : < 100MB
- **Calculs features** : < 1 seconde

## Points de contrôle critiques

### Avant la soutenance

- [ ] **Tous les tests passent** : `poetry run pytest tests/ -v`
- [ ] **MLflow accessible** : Interface web fonctionnelle
- [ ] **API importable** : Pas d'erreur d'import
- [ ] **Evidently fonctionnel** : Migration 0.7+ réussie
- [ ] **Documentation à jour** : README et guides complets

### Éléments manquants identifiés

- [ ] **Evidently Data Drift** : Migration 0.7+ complétée
- [ ] **Tests unitaires** : Structure complète créée
- [ ] **MLflow UI** : Interface documentée et accessible
- [ ] **CI/CD** : Tests sécurité et performance ajoutés

## Validation rapide

### Script automatisé

```bash
./scripts/validate_project.sh
```

### Vérification manuelle

```bash
# 1. Tests
poetry run pytest tests/ -v

# 2. MLflow
./scripts/launch_mlflow.sh

# 3. API
python -c "from api.app import app; print('API OK')"

# 4. Evidently
python -c "import evidently; print(f'Evidently {evidently.__version__}')"
```

## Checklist finale

### Fonctionnalités

- [ ] Score métier avec pondération FN/FP
- [ ] GridSearchCV avec baseline
- [ ] Gestion déséquilibre des classes
- [ ] Feature importance globale et locale
- [ ] Git versioning
- [ ] API production avec GitHub Actions
- [ ] Déploiement cloud (Hugging Face)
- [ ] Data Drift avec Evidently

### Tests

- [ ] Tests unitaires complets
- [ ] Tests d'intégration
- [ ] Tests API
- [ ] Tests de performance
- [ ] Tests de sécurité

### Documentation

- [ ] README principal
- [ ] Guide MLflow
- [ ] Guide de validation
- [ ] Présentation soutenance

## Critères de succès

Le projet est **prêt pour la soutenance** si :

- Tous les tests passent
- MLflow UI accessible
- API fonctionnelle
- Evidently 0.7+ opérationnel
- Documentation complète
- Déploiement Hugging Face fonctionnel

## Prochaines étapes

1. **Exécuter la validation complète**
2. **Corriger les éventuels problèmes**
3. **Préparer la démonstration**
4. **Réviser la présentation**
