# FineTuneFlow — Arborescence Complète des Fichiers

```
finetuneflow/
│
├── docs/                                    # Documentation technique
│   ├── ARCHITECTURE.md                      # Architecture globale + diagrammes
│   ├── DATABASE_SCHEMA.md                   # Schéma DB complet + SQL
│   ├── API_CONTRACT.md                      # Contrat API (tous les endpoints)
│   ├── DATASET_FORMATS.md                   # Formats dataset + validation
│   ├── PROMPTS.md                           # Tous les prompts LLM
│   ├── ML_DEFAULTS.md                       # Hyperparamètres + recommandations
│   ├── ERROR_HANDLING.md                    # Stratégie erreurs + codes
│   ├── SECURITY.md                          # Sécurité (uploads, sandboxing, rate limit)
│   ├── DEVELOPMENT_PLAN.md                  # Plan de dev 24 étapes
│   ├── FRONTEND_SPECS.md                    # Specs UI + wireframes
│   └── FILE_TREE.md                         # Ce fichier
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── alembic.ini
│   ├── pyproject.toml                       # (optionnel) config pytest, ruff, etc.
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                          # FastAPI app, startup, CORS, exception handlers
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py                    # Settings (BaseSettings, .env)
│   │   │   ├── logging.py                   # Configuration structlog
│   │   │   └── exceptions.py                # Exception classes custom
│   │   │
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                      # Base = declarative_base()
│   │   │   ├── session.py                   # Engine, SessionLocal, get_db
│   │   │   ├── models.py                    # Tous les modèles SQLAlchemy
│   │   │   │                                #   Project, File, Chunk, Job,
│   │   │   │                                #   DatasetExample, Run
│   │   │   └── migrations/
│   │   │       ├── env.py
│   │   │       ├── script.py.mako
│   │   │       └── versions/
│   │   │           └── 001_initial_schema.py
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── deps.py                      # Dependencies (get_db, get_project_or_404)
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       ├── health.py                # GET /health
│   │   │       ├── projects.py              # CRUD projects + model resolve
│   │   │       ├── files.py                 # Upload, list, delete files
│   │   │       ├── chunks.py                # Generate chunks, list chunks
│   │   │       ├── dataset.py               # Preview, generate, examples, stats
│   │   │       ├── hardware.py              # GET /hardware/check
│   │   │       ├── training.py              # Start, status, logs/stream, cancel
│   │   │       ├── export.py                # Export, list files, download
│   │   │       ├── jobs.py                  # List jobs, get job, cancel job
│   │   │       └── logs.py                  # SSE log streaming endpoint
│   │   │
│   │   ├── schemas/                         # Pydantic schemas (request/response)
│   │   │   ├── __init__.py
│   │   │   ├── project.py                   # ProjectCreate, ProjectUpdate, ProjectResponse
│   │   │   ├── file.py                      # FileResponse, FileUploadResponse
│   │   │   ├── chunk.py                     # ChunkResponse
│   │   │   ├── dataset.py                   # DatasetExampleResponse, DatasetStats
│   │   │   ├── hardware.py                  # HardwareCheckResponse
│   │   │   ├── training.py                  # TrainingStartRequest, TrainingStatus
│   │   │   ├── export.py                    # ExportFilesResponse
│   │   │   ├── job.py                       # JobResponse
│   │   │   └── common.py                    # PaginatedResponse, ErrorResponse
│   │   │
│   │   ├── services/                        # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── storage.py                   # StorageService (save, read, delete, safe_path)
│   │   │   ├── hf_resolver.py               # Resolve HF model, VRAM estimation
│   │   │   ├── hardware_probe.py            # Detect GPU, CUDA, bnb
│   │   │   ├── recommender.py               # Recommend LoRA/QLoRA based on VRAM
│   │   │   ├── doc_loader.py                # Extract text from PDF/DOCX/TXT
│   │   │   ├── chunking.py                  # Token-based chunking
│   │   │   ├── ollama_client.py             # Ollama Cloud API client (OpenAI-compatible)
│   │   │   ├── dataset_generator.py         # Orchestrate dataset generation (prompts + parse)
│   │   │   ├── dataset_validator.py         # Validate dataset examples
│   │   │   ├── json_repair.py               # Multi-step JSON repair pipeline
│   │   │   ├── token_counter.py             # Count tokens (tiktoken / HF tokenizer)
│   │   │   └── run_report.py                # Generate report.md from run data
│   │   │
│   │   ├── workers/                         # Celery tasks
│   │   │   ├── __init__.py
│   │   │   ├── celery_app.py                # Celery app config (broker, queues, etc.)
│   │   │   ├── tasks_chunking.py            # Task: chunk_documents
│   │   │   ├── tasks_dataset.py             # Tasks: dataset_preview, dataset_generate
│   │   │   ├── tasks_train.py               # Task: train (gpu_queue)
│   │   │   ├── tasks_eval.py                # Task: evaluate
│   │   │   └── tasks_export.py              # Task: export artifacts
│   │   │
│   │   └── ml/                              # Machine learning
│   │       ├── __init__.py
│   │       ├── sft_engine.py                # SFTTrainer wrapper (setup, train, save)
│   │       ├── eval_engine.py               # Evaluation (loss, perplexity, inference samples)
│   │       └── callbacks.py                 # TrainerCallback → Redis Pub/Sub
│   │
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py                      # Fixtures (db session, test client, etc.)
│       ├── test_api/
│       │   ├── test_health.py
│       │   ├── test_projects.py
│       │   ├── test_files.py
│       │   ├── test_dataset.py
│       │   ├── test_training.py
│       │   └── test_export.py
│       ├── test_services/
│       │   ├── test_storage.py
│       │   ├── test_doc_loader.py
│       │   ├── test_chunking.py
│       │   ├── test_json_repair.py
│       │   ├── test_dataset_validator.py
│       │   ├── test_ollama_client.py
│       │   └── test_recommender.py
│       └── test_workers/
│           ├── test_tasks_dataset.py
│           └── test_tasks_train.py
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── next.config.js
│   ├── postcss.config.js
│   │
│   ├── app/
│   │   ├── layout.tsx                       # Root layout (fonts, providers)
│   │   ├── page.tsx                         # Dashboard (liste projets)
│   │   ├── globals.css                      # Tailwind + custom styles
│   │   │
│   │   └── projects/
│   │       ├── new/
│   │       │   └── page.tsx                 # Création projet
│   │       └── [id]/
│   │           ├── page.tsx                 # Détail projet (redirect wizard)
│   │           └── wizard/
│   │               ├── page.tsx             # Wizard layout + navigation
│   │               └── steps/
│   │                   ├── model-step.tsx
│   │                   ├── task-step.tsx
│   │                   ├── data-step.tsx
│   │                   ├── preview-step.tsx
│   │                   ├── review-step.tsx
│   │                   ├── hardware-step.tsx
│   │                   ├── train-step.tsx
│   │                   └── export-step.tsx
│   │
│   ├── components/
│   │   ├── ui/                              # shadcn/ui components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── dialog.tsx
│   │   │   ├── input.tsx
│   │   │   ├── select.tsx
│   │   │   ├── table.tsx
│   │   │   ├── badge.tsx
│   │   │   ├── progress.tsx
│   │   │   ├── toast.tsx
│   │   │   ├── tabs.tsx
│   │   │   └── ...
│   │   ├── file-upload-zone.tsx             # Drag & drop upload
│   │   ├── job-progress.tsx                 # Barre progression + polling
│   │   ├── data-table.tsx                   # Tableau paginé + filtres
│   │   ├── status-badge.tsx                 # Badge de statut coloré
│   │   ├── loss-chart.tsx                   # Graphique loss (recharts)
│   │   ├── log-stream.tsx                   # Affichage logs SSE temps réel
│   │   ├── error-toast.tsx                  # Notification erreur
│   │   ├── confirm-dialog.tsx               # Dialog confirmation
│   │   ├── hyperparam-form.tsx              # Formulaire hyperparamètres
│   │   └── project-card.tsx                 # Carte projet (dashboard)
│   │
│   └── lib/
│       ├── api.ts                           # Client API (fetch wrapper)
│       ├── types.ts                         # Types TypeScript (interfaces)
│       ├── store.ts                         # Zustand store (wizard state)
│       ├── constants.ts                     # Constantes (steps, defaults)
│       └── utils.ts                         # Helpers (format date, size, etc.)
│
├── storage/                                 # Local file storage (dev)
│   └── .gitkeep
│
├── docker-compose.yml
├── .env.example
├── .env                                     # (gitignored)
├── .gitignore
└── README.md
```

## Comptage de fichiers

| Dossier | Fichiers | Description |
|---------|----------|-------------|
| `docs/` | 11 | Documentation technique |
| `backend/app/core/` | 4 | Config, logging, exceptions |
| `backend/app/db/` | 5 + migrations | Modèles DB, session, migrations |
| `backend/app/api/routes/` | 11 | Endpoints API |
| `backend/app/schemas/` | 10 | Pydantic schemas |
| `backend/app/services/` | 12 | Business logic |
| `backend/app/workers/` | 6 | Celery tasks |
| `backend/app/ml/` | 4 | ML engine |
| `backend/tests/` | 15+ | Tests |
| `frontend/app/` | 12 | Pages Next.js |
| `frontend/components/` | 20+ | Composants React |
| `frontend/lib/` | 5 | Utilitaires |
| Racine | 5 | Config (docker, env, gitignore, readme) |
| **Total** | **~120** | |
