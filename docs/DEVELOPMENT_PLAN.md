# FineTuneFlow — Plan de Développement

## Vue d'ensemble

24 étapes organisées en 5 phases. Chaque étape a un **livrable vérifiable**.
Le frontend est intégré à chaque phase (pas à la fin).

**Temps estimé total MVP :** 6-8 semaines (1 développeur)

---

## Phase 0 — Setup Infrastructure (3-4 jours)

### Étape 1 : Repo + Docker Compose + Healthchecks
**Objectif :** `docker compose up` → tous les services green

**Tâches :**
- [ ] Initialiser la structure des dossiers (`backend/`, `frontend/`, `storage/`, `docs/`)
- [ ] Créer `docker-compose.yml` avec : PostgreSQL 16, Redis 7, backend, worker, frontend
- [ ] Ajouter healthchecks sur postgres et redis
- [ ] Créer `.env.example` et `.env`
- [ ] Créer `.gitignore` complet
- [ ] Créer `backend/Dockerfile`
- [ ] Créer `frontend/Dockerfile`
- [ ] Créer `backend/requirements.txt`
- [ ] Créer `frontend/package.json`

**Vérification :** `docker compose up -d && docker compose ps` → tous les containers `healthy`

---

### Étape 2 : FastAPI skeleton + /health + CORS
**Objectif :** `curl localhost:8000/health` → JSON OK

**Tâches :**
- [ ] `backend/app/main.py` — FastAPI app avec CORS, exception handlers
- [ ] `backend/app/core/config.py` — Settings Pydantic (BaseSettings, .env)
- [ ] `backend/app/core/logging.py` — Configuration structlog
- [ ] `backend/app/core/exceptions.py` — Classes d'exception custom
- [ ] Route `GET /health` avec check DB + Redis
- [ ] Rate limiter global (slowapi)

**Vérification :** `curl http://localhost:8000/health` → `{"status": "ok", "db": "connected", "redis": "connected"}`

---

### Étape 3 : DB Schema + Alembic migrations
**Objectif :** `alembic upgrade head` → tables créées

**Tâches :**
- [ ] `backend/app/db/base.py` — Base SQLAlchemy
- [ ] `backend/app/db/session.py` — SessionLocal, engine, get_db dependency
- [ ] `backend/app/db/models.py` — Tous les modèles (projects, files, chunks, jobs, dataset_examples, runs)
- [ ] Initialiser Alembic (`alembic init`)
- [ ] Configurer `alembic/env.py` pour SQLAlchemy
- [ ] Générer migration initiale (`alembic revision --autogenerate`)
- [ ] Tester `upgrade` et `downgrade`

**Vérification :** `alembic upgrade head && psql -c "\dt"` → voir les 6 tables

---

### Étape 4 : Celery app + Redis + task de test
**Objectif :** Celery worker démarre et exécute une task de test

**Tâches :**
- [ ] `backend/app/workers/celery_app.py` — Configuration Celery (broker, backend, queues)
- [ ] Définir 2 queues : `default` et `gpu_queue` (concurrency=1)
- [ ] Task de test `ping` qui retourne "pong"
- [ ] Configurer retry defaults
- [ ] Tester avec `celery call`

**Vérification :** `celery -A app.workers.celery_app call app.workers.celery_app.ping` → retourne "pong"

---

## Phase 1 — Upload & Gestion Projets (3-4 jours)

### Étape 5 : Storage service + modèle files
**Objectif :** Service capable de sauvegarder/lire des fichiers avec sandboxing

**Tâches :**
- [ ] `backend/app/services/storage.py` — StorageService (save, read, delete, list, safe_path)
- [ ] Sanitisation des noms de fichiers (path traversal protection)
- [ ] Validation taille max
- [ ] Création auto des sous-dossiers (`raw/`, `datasets/`, `artifacts/`, etc.)
- [ ] Tests unitaires

**Vérification :** Tests passent, y compris les cas de path traversal

---

### Étape 6 : API Projects CRUD + Files upload
**Objectif :** Créer un projet, uploader un PDF via curl

**Tâches :**
- [ ] `backend/app/api/routes/projects.py` — CRUD complet (POST, GET list, GET detail, PATCH, DELETE)
- [ ] `backend/app/api/routes/files.py` — POST upload (multipart), GET list, DELETE
- [ ] `backend/app/api/deps.py` — Dependencies (get_db, get_project_or_404)
- [ ] Validation MIME type + taille + extension
- [ ] Schémas Pydantic (request/response) pour projects et files
- [ ] Rate limiting sur upload

**Vérification :**
```bash
curl -X POST localhost:8000/projects -d '{"name":"test","task_type":"instruction"}'
curl -X POST localhost:8000/projects/{id}/files/upload -F "files=@document.pdf" -F "kind=raw_doc"
curl localhost:8000/projects/{id}/files
```

---

### Étape 7 : Frontend — Dashboard + Création projet + Upload
**Objectif :** UI fonctionnelle pour créer un projet et uploader des fichiers

**Tâches :**
- [ ] `frontend/app/page.tsx` — Page d'accueil / dashboard (liste projets)
- [ ] `frontend/app/projects/new/page.tsx` — Formulaire création projet
- [ ] `frontend/app/projects/[id]/page.tsx` — Détail projet (hub du wizard)
- [ ] `frontend/lib/api.ts` — Client API (fetch wrapper)
- [ ] `frontend/components/file-upload.tsx` — Composant upload drag & drop
- [ ] Setup Zustand store pour l'état global
- [ ] Setup Tailwind + shadcn/ui

**Vérification :** Navigateur → créer projet → uploader PDF → voir le fichier dans la liste

---

## Phase 2 — Dataset Generation (5-7 jours)

### Étape 8 : Doc loader + Chunking
**Objectif :** PDF → chunks en DB avec comptage de tokens

**Tâches :**
- [ ] `backend/app/services/doc_loader.py` — Extraction texte PDF (PyPDF2), DOCX (python-docx), TXT
- [ ] `backend/app/services/chunking.py` — Chunking par tokens (tiktoken/transformers tokenizer)
  - Paramètres : chunk_size (512), overlap (50)
  - Gestion des chunks trop petits (fusion) et trop grands (split)
- [ ] `backend/app/workers/tasks_dataset.py` — Task Celery `chunk_documents`
- [ ] Route `POST /projects/{id}/chunks/generate`
- [ ] Route `GET /projects/{id}/chunks`
- [ ] Tests unitaires pour chaque type de doc

**Vérification :** Upload PDF → POST chunks/generate → GET chunks → voir les chunks avec token_count

---

### Étape 9 : Client Ollama Cloud
**Objectif :** Appel Ollama Cloud → réponse parsée

**Tâches :**
- [ ] `backend/app/services/ollama_client.py` — Client HTTP (httpx async)
  - Endpoint OpenAI-compatible (`/v1/chat/completions`)
  - Auth via API key
  - Retry avec tenacity (3 tentatives, backoff exponentiel)
  - Timeout configurable (60s)
  - Parsing de la réponse
- [ ] Validation de la clé API au démarrage
- [ ] Test d'intégration (mock ou vrai appel)

**Vérification :** Script de test → appel Ollama → réponse JSON parsée

---

### Étape 10 : Dataset preview job (10 exemples)
**Objectif :** POST preview → 10 exemples valides en DB

**Tâches :**
- [ ] `backend/app/services/dataset_schema.py` — Schémas de validation des exemples
- [ ] `backend/app/workers/tasks_dataset.py` — Task `dataset_preview`
  - Sélection de 3-5 chunks représentatifs
  - Appel Ollama avec prompts (instruction ou Q&A selon task_type)
  - Validation + repair
  - Insert en DB (dataset_examples, split=preview)
  - Update job progress
- [ ] Route `POST /projects/{id}/dataset/preview`
- [ ] Route `GET /projects/{id}/dataset/examples` (paginé)

**Vérification :** POST preview → attendre job → GET examples?split=preview → 10 exemples valides

---

### Étape 11 : JSON repair pipeline
**Objectif :** Réponses LLM mal formées → JSON valide

**Tâches :**
- [ ] `backend/app/services/json_repair.py` — Pipeline de repair
  1. Parse strict
  2. Strip markdown fences
  3. Regex extract JSON objects
  4. Fix trailing commas, missing brackets
  5. Fallback LLM repair (1 tentative)
- [ ] Tests unitaires avec cas réels cassés

**Vérification :** Tests passent sur 10+ cas de JSON cassé

---

### Étape 12 : Dataset génération complète + dedup + export JSONL
**Objectif :** Job full → train.jsonl + eval.jsonl

**Tâches :**
- [ ] Task `dataset_generate` — Batch processing de tous les chunks
  - Progress tracking (update progress_pct)
  - Dedup par hash SHA-256
  - Split train/eval (90/10)
  - Token counting
  - Export train.jsonl et eval.jsonl sur disque
- [ ] Route `POST /projects/{id}/dataset/generate`
- [ ] Route `GET /projects/{id}/dataset/stats`

**Vérification :** POST generate → job 100% → train.jsonl et eval.jsonl existent → stats cohérentes

---

### Étape 13 : Frontend — Wizard steps (Task, Data, Preview, Review)
**Objectif :** Wizard fonctionnel jusqu'au review du dataset

**Tâches :**
- [ ] `frontend/app/projects/[id]/wizard/page.tsx` — Layout wizard multi-étapes
- [ ] `model-step.tsx` — Champ modèle HF (autocomplete optionnel)
- [ ] `task-step.tsx` — Choix Instruction / Q&A
- [ ] `data-step.tsx` — Upload docs ou dataset
- [ ] `preview-step.tsx` — Lancer preview + afficher 10 exemples
- [ ] `review-step.tsx` — Tableau paginé des exemples, stats, bouton supprimer
- [ ] Composant de progression des jobs
- [ ] Navigation entre steps (prev/next, état)

**Vérification :** Parcourir le wizard complet de Model → Review dans le navigateur

---

## Phase 3 — Training (5-7 jours)

### Étape 14 : HF model resolver + VRAM estimation
**Objectif :** Résolution modèle HF → metadata + estimation VRAM

**Tâches :**
- [ ] `backend/app/services/hf_resolver.py`
  - Appel HF Hub API (`huggingface_hub.model_info()`)
  - Extraction : architecture, params, config
  - Estimation VRAM (FP16 et 4-bit) basée sur les params
  - Cache résultats en DB (`projects.model_info`)
- [ ] Route `POST /projects/{id}/model/resolve`
- [ ] Gestion erreurs : modèle non trouvé, gated, etc.

**Vérification :** POST resolve avec "meta-llama/Llama-3.1-8B" → JSON avec estimation VRAM

---

### Étape 15 : Hardware probe endpoint
**Objectif :** GET /hardware/check → JSON GPU info

**Tâches :**
- [ ] `backend/app/services/hardware_probe.py` — Détecter GPU, CUDA, bnb
- [ ] `backend/app/services/recommender.py` — Recommandation LoRA/QLoRA
- [ ] Route `GET /hardware/check`
- [ ] Route `GET /hardware/check?model_id=xxx` — avec estimation pour un modèle spécifique

**Vérification :** GET hardware/check → JSON avec gpu_name, vram, recommended_method

---

### Étape 16 : SFT Engine (TRL + PEFT)
**Objectif :** Training sur dummy dataset → adapter sauvé

**Tâches :**
- [ ] `backend/app/ml/sft_engine.py` — Classe SFTEngine
  - `setup_model()` — Chargement avec quantification si QLoRA
  - `setup_tokenizer()` — Config padding, special tokens
  - `load_dataset()` — Lire train.jsonl + eval.jsonl
  - `format_examples()` — Formatting function pour SFTTrainer
  - `train()` — Lancer SFTTrainer.train()
  - `save_adapter()` — Sauvegarder LoRA adapter
- [ ] Callback Redis Pub/Sub pour logs temps réel
- [ ] Early stopping callback
- [ ] Tests avec un petit modèle (ex: TinyLlama)

**Vérification :** Lancer training local avec TinyLlama + 50 exemples → adapter sauvegardé

---

### Étape 17 : Job training Celery + logs Redis Pub/Sub
**Objectif :** Job end-to-end avec logs

**Tâches :**
- [ ] `backend/app/workers/tasks_train.py` — Task `train`
  - Hardware check
  - Recommandation méthode
  - Appel SFTEngine
  - Update job progress
  - Gestion OOM (catch + cleanup)
  - Sauvegarde run en DB (hyperparams, metrics)
- [ ] Publication logs via Redis Pub/Sub (`logs:{job_id}`)
- [ ] Route `POST /projects/{id}/train/start` — avec vérif pas de training concurrent
- [ ] Route `GET /projects/{id}/train/status`
- [ ] Route `GET /projects/{id}/train/logs/stream` — SSE
- [ ] Route `POST /projects/{id}/train/cancel`

**Vérification :** POST train/start → SSE stream montre les logs → job success → adapter sur disque

---

### Étape 18 : Frontend — Hardware, Train, Logs streaming
**Objectif :** UI training avec logs live

**Tâches :**
- [ ] `hardware-step.tsx` — Affiche GPU info + recommandation + warning si pas de GPU
- [ ] `train-step.tsx` — Configuration hyperparams (avec defaults) + bouton Start
- [ ] Composant `training-logs.tsx` — EventSource SSE, affichage temps réel
- [ ] Composant `loss-chart.tsx` — Graphique train/eval loss (recharts ou chart.js)
- [ ] État du training (progress bar, métriques courantes)

**Vérification :** Lancer training depuis l'UI → voir les logs se mettre à jour en temps réel

---

## Phase 4 — Export & Finitions (3-4 jours)

### Étape 19 : Eval engine
**Objectif :** Métriques d'évaluation calculées

**Tâches :**
- [ ] `backend/app/ml/eval_engine.py`
  - Calcul eval_loss (déjà dans SFTTrainer)
  - Perplexité
  - Inférence sur 5-10 exemples eval (comparaison avant/après)
  - Sauvegarde metrics en DB (runs.metrics)
- [ ] `backend/app/workers/tasks_eval.py` — Task eval (peut être intégré au train)

**Vérification :** Après training → runs.metrics contient eval_loss, perplexity, samples

---

### Étape 20 : Export job + report
**Objectif :** Zip téléchargeable avec adapter + report

**Tâches :**
- [ ] `backend/app/workers/tasks_export.py` — Task export
  - Copie adapter files
  - Génère `report.md` (template Jinja2)
  - Génère `metrics.json`
  - Génère `training_config.json`
  - Crée zip final
- [ ] `backend/app/services/run_report.py` — Générateur de rapport
- [ ] Routes export : POST, GET files, GET download

**Vérification :** POST export → GET download → zip contient adapter + report + metrics

---

### Étape 21 : Frontend — Export + Téléchargement
**Objectif :** Wizard complet de bout en bout

**Tâches :**
- [ ] `export-step.tsx` — Résumé du training, métriques, graphiques
- [ ] Bouton Export → téléchargement zip
- [ ] Liste des fichiers exportés
- [ ] Affichage du rapport (markdown rendu)

**Vérification :** Parcourir tout le wizard : Model → Export → télécharger le zip

---

## Phase 5 — Polish & Tests (3-4 jours)

### Étape 22 : Tests d'intégration
**Objectif :** Tests automatisés sur le flow complet

**Tâches :**
- [ ] Tests API (pytest + httpx)
- [ ] Tests services (unitaires)
- [ ] Tests workers (avec Celery en mode eager)
- [ ] Test intégration : upload → dataset → training (avec modèle tiny)
- [ ] GitHub Actions CI (optionnel mais recommandé)

**Vérification :** `pytest` → tous les tests green

---

### Étape 23 : Error handling robuste + edge cases
**Objectif :** Scénarios d'échec testés et gérés

**Tâches :**
- [ ] Tester : Ollama API down → retry → échec graceful
- [ ] Tester : Upload fichier corrompu → erreur claire
- [ ] Tester : Training OOM → catch + message + cleanup
- [ ] Tester : Annulation de job en cours
- [ ] Tester : Redémarrage serveur pendant un job
- [ ] Frontend : affichage propre des erreurs (toast notifications)

**Vérification :** Chaque scénario d'erreur est testé et donne un message clair

---

### Étape 24 : README + Documentation
**Objectif :** Projet documenté et prêt à l'usage

**Tâches :**
- [ ] `README.md` complet (description, screenshots, installation, usage)
- [ ] Guide d'installation (Docker + dev local)
- [ ] Guide de configuration (Ollama API key, etc.)
- [ ] Troubleshooting commun (GPU non détecté, OOM, etc.)
- [ ] Nettoyer le code (dead code, TODO, etc.)

**Vérification :** Un développeur peut cloner le repo et lancer le projet en suivant le README

---

## Dépendances entre étapes

```
Phase 0: 1 → 2 → 3 → 4 (séquentiel)
Phase 1: 5 → 6 → 7 (séquentiel)
Phase 2: 8 → 9 → 10 → 11 → 12 → 13
         (8 et 9 peuvent être en parallèle)
         (11 peut commencer avec 10)
Phase 3: 14 + 15 (parallèle) → 16 → 17 → 18
Phase 4: 19 → 20 → 21
Phase 5: 22 + 23 (parallèle) → 24
```

## Risques et mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Ollama Cloud API change | Bloque la génération | Client abstrait, easy to swap |
| OOM pendant training | Mauvaise UX | Détection + suggestion QLoRA/réduction |
| Modèle HF gated | Bloque le training | Message clair + lien vers la page HF |
| Qualité dataset basse | Training inutile | Preview + review + stats + dedup |
| Réponses LLM invalides | Dataset incomplet | Pipeline repair multi-étapes |
