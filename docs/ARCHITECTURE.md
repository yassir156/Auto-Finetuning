# FineTuneFlow — Architecture Globale (MVP)

## 1. Vue d'ensemble

FineTuneFlow est une application **locale** permettant à un utilisateur de :
1. Uploader des documents (PDF, DOCX, TXT) ou un dataset existant
2. Générer automatiquement un dataset de fine-tuning via Ollama Cloud API
3. Fine-tuner un modèle HuggingFace avec LoRA/QLoRA
4. Exporter les artifacts (adapter, metrics, report)

**Tout tourne en local sur le PC de l'utilisateur.** Pas d'auth nécessaire.

## 2. Diagramme d'architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    MACHINE LOCALE USER                        │
│                                                              │
│  ┌─────────────────┐         ┌──────────────────────────┐   │
│  │  Frontend        │  REST   │  FastAPI Backend          │   │
│  │  Next.js 14      │────────▶│  Port 8000                │   │
│  │  Port 3000       │◀────────│                          │   │
│  │                  │  + SSE   │  ┌──────────────────┐   │   │
│  │  Wizard UI       │         │  │  SQLAlchemy ORM   │   │   │
│  │  - Model step    │         │  │  Alembic migrate  │   │   │
│  │  - Task step     │         │  └────────┬─────────┘   │   │
│  │  - Data step     │         │           │              │   │
│  │  - Preview step  │         │           ▼              │   │
│  │  - Review step   │         │  ┌──────────────────┐   │   │
│  │  - Hardware step │         │  │  PostgreSQL 16    │   │   │
│  │  - Train step    │         │  │  Port 5432        │   │   │
│  │  - Export step   │         │  └──────────────────┘   │   │
│  └─────────────────┘         │                          │   │
│                               │  Enqueue ──▶ Celery     │   │
│                               └──────────┬───────────────┘   │
│                                          │                    │
│                                          ▼                    │
│                               ┌──────────────────────────┐   │
│                               │  Redis 7                  │   │
│                               │  Port 6379                │   │
│                               │  - Celery broker          │   │
│                               │  - Pub/Sub logs           │   │
│                               └──────────┬───────────────┘   │
│                                          │                    │
│                                          ▼                    │
│                               ┌──────────────────────────┐   │
│                               │  Celery Worker            │   │
│                               │                          │   │
│                               │  Queues:                 │   │
│                               │  - default (dataset gen) │   │
│                               │  - gpu (train, concur=1) │   │
│                               │                          │   │
│                               │  Tasks:                  │   │
│                               │  - dataset_preview       │   │
│                               │  - dataset_generate      │   │
│                               │  - train                 │   │
│                               │  - eval                  │   │
│                               │  - export                │   │
│                               └──────────┬───────────────┘   │
│                                          │                    │
│                          ┌───────────────┼───────────────┐    │
│                          ▼               ▼               ▼    │
│                   ┌────────────┐  ┌────────────┐  ┌────────┐ │
│                   │Ollama Cloud│  │HF Trans+TRL│  │Local   │ │
│                   │API (remote)│  │+PEFT+bnb   │  │Storage │ │
│                   └────────────┘  └────────────┘  └────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 3. Composants

### 3.1 Frontend (Next.js 14)
- **Framework** : Next.js 14 App Router + React 18
- **State** : Zustand (store global wizard)
- **UI** : Tailwind CSS + shadcn/ui
- **Communication** : fetch REST + EventSource SSE
- **Pages** :
  - `/` — Dashboard (liste projets)
  - `/projects/[id]/wizard` — Wizard multi-étapes

### 3.2 Backend (FastAPI)
- **Framework** : FastAPI 0.110+
- **ORM** : SQLAlchemy 2.0 (async)
- **Migrations** : Alembic
- **Validation** : Pydantic v2
- **CORS** : Configuré pour localhost:3000
- **Rate limiting** : slowapi (par IP)
- **Upload** : Validation taille + type MIME + sandboxing path

### 3.3 Base de données (PostgreSQL 16)
- Tables : `projects`, `files`, `chunks`, `jobs`, `dataset_examples`, `runs`
- Types enum PostgreSQL pour les statuts
- JSONB pour config flexible (hyperparams, metrics)

### 3.4 Queue de tâches (Celery + Redis)
- **Broker** : Redis
- **Backend résultat** : Redis
- **Queues** :
  - `default` : tâches dataset (parallélisable)
  - `gpu_queue` : tâches training (concurrency=1 pour éviter OOM)
- **Retry** : autoretry avec backoff exponentiel
- **Logs** : publication Redis Pub/Sub → SSE

### 3.5 Ollama Cloud API
- **Endpoint** : `https://api.ollama.com/v1/chat/completions` (OpenAI-compatible)
- **Auth** : API key fournie par l'utilisateur (créée sur ollama.com)
- **Modèle par défaut** : configurable via `.env` (ex: `llama3.1:70b`)
- **Usage** : génération de dataset uniquement (pas pour le training)

### 3.6 ML Stack
- `transformers` — chargement modèles HF
- `trl` (SFTTrainer) — fine-tuning supervisé
- `peft` — LoRA / QLoRA adapters
- `bitsandbytes` — quantification 4-bit (QLoRA)
- `accelerate` — device mapping automatique
- `datasets` — chargement données

### 3.7 Storage local
```
./storage/
  {project_uuid}/
    raw/           # fichiers uploadés originaux
    chunks/        # chunks extraits (optionnel, principalement en DB)
    datasets/      # train.jsonl, eval.jsonl
    artifacts/     # adapter, config, tokenizer
    logs/          # training logs
    exports/       # report.md, metrics.json, zip final
```

## 4. Flux de données

### 4.1 Upload → Dataset Preview
```
User upload PDF ──▶ Backend valide (taille, type, path) 
  ──▶ Sauvegarde ./storage/{project}/raw/
  ──▶ Insert DB (files table)
  ──▶ Doc loader (PyPDF2/python-docx) 
  ──▶ Chunking (token-based, ~512 tokens, overlap 50)
  ──▶ Insert DB (chunks table)
  ──▶ Celery: dataset_preview
      ──▶ Prend 3-5 chunks représentatifs
      ──▶ Appel Ollama Cloud (2-3 exemples par chunk)
      ──▶ Validation JSONL
      ──▶ Repair si invalide (programmatique d'abord, puis LLM)
      ──▶ Insert DB (dataset_examples, split=preview)
  ──▶ Backend retourne les exemples preview
```

### 4.2 Dataset Generation (complet)
```
Celery: dataset_generate
  ──▶ Pour chaque chunk (batch de 5):
      ──▶ Appel Ollama Cloud ({N} exemples par chunk)
      ──▶ Validation + repair
      ──▶ Dedup (hash exact + fuzzy optionnel)
      ──▶ Token counting (vérif max_seq_length)
      ──▶ Insert DB (dataset_examples, split=train ou eval)
      ──▶ Update job progress_pct
  ──▶ Export train.jsonl + eval.jsonl sur disque
  ──▶ Stats: total exemples, rejetés, distribution
```

### 4.3 Training
```
Celery (gpu_queue, concurrency=1):
  ──▶ Hardware probe
  ──▶ Recommandation LoRA/QLoRA
  ──▶ Chargement modèle (4-bit si QLoRA)
  ──▶ Chargement dataset (train.jsonl)
  ──▶ SFTTrainer.train()
      ──▶ Callback: log chaque step → Redis Pub/Sub
      ──▶ Checkpoint toutes les N steps
      ──▶ Early stopping si eval_loss diverge
  ──▶ Sauvegarde adapter → ./storage/{project}/artifacts/
  ──▶ Eval sur eval set
  ──▶ Sauvegarde metrics
```

### 4.4 Export
```
Celery: export
  ──▶ Copie adapter files
  ──▶ Génère report.md (métriques, config, exemples)
  ──▶ Génère metrics.json
  ──▶ Génère config.json (hyperparams utilisés)
  ──▶ Zip le tout
  ──▶ Disponible au téléchargement
```

## 5. Communication inter-services

| Source | Destination | Protocole | Usage |
|--------|-------------|-----------|-------|
| Frontend | Backend | REST (HTTP) | CRUD, actions |
| Frontend | Backend | SSE (HTTP) | Logs training temps réel |
| Backend | Celery | Redis (AMQP) | Enqueue tâches |
| Worker | Redis | Pub/Sub | Publication logs |
| Backend | Redis | Pub/Sub (subscribe) | Relay logs → SSE |
| Worker | Ollama Cloud | HTTPS | Génération dataset |
| Worker | HF Hub | HTTPS | Download modèles |
| Backend/Worker | PostgreSQL | TCP | Persistance |
| Backend/Worker | Storage local | Filesystem | Fichiers |

## 6. Contraintes et décisions

| Décision | Justification |
|----------|---------------|
| Pas d'auth | App locale, un seul utilisateur |
| Ollama Cloud (pas local) | Évite de télécharger un gros modèle pour la génération |
| Storage local (pas S3) | MVP local, simplicité |
| Redis Pub/Sub pour logs | Plus simple que WebSocket, Celery utilise déjà Redis |
| GPU queue concurrency=1 | Un seul GPU, éviter OOM |
| Alembic pour migrations | Standard SQLAlchemy, versionning DB |
| Zustand (pas Redux) | Plus léger, suffisant pour un wizard |
