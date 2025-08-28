# Rules Pack — Qualité des Règles

Ce dépôt contient le Rules Pack (source unique des règles) et leurs mécanismes d’activation/CI.

## QA des Règles (local & CI)

Consultez la section dédiée dans `AGENTS.md` pour les commandes et critères:

- Règles — QA locale/CI: voir `AGENTS.md` (section “Règles — QA locale/CI”)
- Lint métadonnées: `python docs/scripts/rules_meta_lint.py`
- Scan duplication adaptateurs: `bash docs/scripts/scan_adapters_dup.sh`
- CI: workflow `.github/workflows/rules-ci.yml` (à définir comme required check)

Politique de source unique: ne pas dupliquer le contenu normatif (MUST/SHOULD) hors `docs/rules/**`.

