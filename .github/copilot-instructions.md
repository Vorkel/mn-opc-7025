# Rules Pack

**Source de vérité:**
- /docs/rules/rules.index.yaml
- /docs/rules/**/rule-<slug>.md

**Toujours actives:**
rules: ai-governance-style, ai-stepwise, ai-halt-on-uncertainty, ai-code-run-build

**Exigences conversationnelles:**
- En tête: afficher `rules: …`.
- Boucle code → run → build jusqu’au vert ; logs d’erreurs synthétisés ; proposer correctifs ciblés.
- Si incertitude: ≤3 questions (stop & clarify).
- Style pro, concis, sans emoji.

**Projection:**
- Au lieu de résumer les règles, ouvre le fichier canonique correspondant et cite ses MUST/SHOULD.

**Configuration:**

- Source: /docs/rules/rules.activation.yaml
- Toujours actives: ai-governance-style, ai-stepwise, ai-halt-on-uncertainty, ai-code-run-build
- Logs: .rules/logs/activation_log.jsonl
- Overrides: rules.local.yaml

**Utilisation:**
Copilot utilise ce fichier pour comprendre le système d'auto-activation
et appliquer les règles contextuelles appropriées.
