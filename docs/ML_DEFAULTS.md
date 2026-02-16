# FineTuneFlow — ML Defaults & Recommandations

## 1. Hyperparamètres par défaut

### 1.1 Training (SFTTrainer)

```python
TRAINING_DEFAULTS = {
    # Training
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "fp16": False,           # True si GPU supporte, et pas bf16
    "bf16": True,            # True si GPU supporte (Ampere+)
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",  # pour QLoRA
    
    # Logging & Checkpointing
    "logging_steps": 10,
    "eval_steps": 100,         # évaluation toutes les 100 steps
    "save_steps": 200,         # checkpoint toutes les 200 steps
    "save_total_limit": 3,     # garder max 3 checkpoints
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    
    # Early stopping
    "early_stopping_patience": 3,  # stop après 3 evals sans amélioration
}
```

### 1.2 LoRA

```python
LORA_DEFAULTS = {
    "r": 16,                    # rang de la décomposition
    "lora_alpha": 32,           # scaling factor (alpha/r = 2)
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "auto",   # PEFT détecte automatiquement (q_proj, v_proj, etc.)
}
```

### 1.3 QLoRA (4-bit)

```python
QLORA_DEFAULTS = {
    **LORA_DEFAULTS,
    # BitsAndBytes config
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",   # ou float16
    "bnb_4bit_quant_type": "nf4",           # NormalFloat4
    "bnb_4bit_use_double_quant": True,      # double quantization (économise ~0.4GB)
}
```

## 2. Recommandation méthode (heuristique)

### 2.1 Logique de décision

```python
def recommend_method(
    vram_gb: float,
    model_params_b: float,  # en milliards
    cuda_available: bool,
    bnb_available: bool
) -> dict:
    """
    Retourne la méthode recommandée et les raisons.
    """
    if not cuda_available:
        return {
            "method": None,
            "reason": "No CUDA GPU detected. Cannot fine-tune locally.",
            "can_train": False,
            "suggestions": ["Use a cloud GPU (Colab, RunPod, Lambda)"]
        }
    
    # Estimation VRAM nécessaire
    vram_fp16 = model_params_b * 2 * 1.2    # params × 2 bytes × overhead
    vram_4bit = model_params_b * 0.5 * 1.3  # params × 0.5 bytes × overhead
    # LoRA overhead: ~10-15% pour l'optimizer et les gradients des adapters
    vram_lora = vram_fp16 * 1.15
    vram_qlora = vram_4bit * 1.3  # plus d'overhead relatif pour optimizer
    
    if vram_gb < 6:
        return {
            "method": None,
            "reason": f"Only {vram_gb}GB VRAM. Minimum 6GB required for QLoRA with small models.",
            "can_train": False,
            "suggestions": ["Use a GPU with more VRAM", "Try a smaller model (< 3B params)"]
        }
    
    if vram_gb < 12:
        if vram_qlora <= vram_gb * 0.85:  # 85% margin
            return {
                "method": "qlora",
                "reason": f"{vram_gb}GB VRAM. QLoRA 4-bit recommended to fit in memory.",
                "can_train": True,
                "estimated_vram_usage_gb": round(vram_qlora, 1),
                "suggestions": ["Reduce batch_size if OOM", "Reduce max_seq_length"]
            }
        return {
            "method": "qlora",
            "reason": f"{vram_gb}GB VRAM might be tight for {model_params_b}B model.",
            "can_train": True,
            "warnings": ["High OOM risk. Reduce batch_size to 1 and max_seq_length to 1024."],
            "estimated_vram_usage_gb": round(vram_qlora, 1)
        }
    
    if vram_gb < 24:
        if bnb_available and vram_lora > vram_gb * 0.85:
            return {
                "method": "qlora",
                "reason": f"{vram_gb}GB VRAM. Model too large for LoRA FP16, using QLoRA.",
                "can_train": True,
                "estimated_vram_usage_gb": round(vram_qlora, 1)
            }
        return {
            "method": "lora",
            "reason": f"{vram_gb}GB VRAM. LoRA FP16 fits comfortably.",
            "can_train": True,
            "estimated_vram_usage_gb": round(vram_lora, 1),
            "alternative": "qlora (saves ~50% VRAM)"
        }
    
    # >= 24 GB
    return {
        "method": "lora",
        "reason": f"{vram_gb}GB VRAM. LoRA FP16 recommended for best quality.",
        "can_train": True,
        "estimated_vram_usage_gb": round(vram_lora, 1),
        "alternative": "qlora (if you want to save VRAM for larger batch sizes)"
    }
```

### 2.2 Matrice de décision (résumé)

| VRAM | Modèle ≤ 3B | Modèle 7-8B | Modèle 13B | Modèle 70B |
|------|-------------|-------------|------------|------------|
| < 6 GB | ❌ | ❌ | ❌ | ❌ |
| 6-8 GB | QLoRA ⚠️ | QLoRA ⚠️ | ❌ | ❌ |
| 8-12 GB | QLoRA ✅ | QLoRA ⚠️ | ❌ | ❌ |
| 12-16 GB | LoRA ✅ | QLoRA ✅ | QLoRA ⚠️ | ❌ |
| 16-24 GB | LoRA ✅ | LoRA ✅ | QLoRA ✅ | ❌ |
| 24-48 GB | LoRA ✅ | LoRA ✅ | LoRA ✅ | QLoRA ⚠️ |
| 48+ GB | LoRA ✅ | LoRA ✅ | LoRA ✅ | QLoRA ✅ |

✅ = confortable | ⚠️ = possible avec réductions (batch=1, seq_len=1024) | ❌ = impossible

### 2.3 Ajustements automatiques si VRAM limité

```python
VRAM_TIGHT_ADJUSTMENTS = {
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,  # compenser batch petit
    "max_seq_length": 1024,            # réduire
    "gradient_checkpointing": True,    # obligatoire
}
```

## 3. Hardware Probe — Détail

### 3.1 Checks effectués

```python
def probe_hardware() -> dict:
    result = {}
    
    # 1. nvidia-smi
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"])
        # parse CSV → gpu_name, vram_total, vram_free, driver
        result["has_nvidia_smi"] = True
    except FileNotFoundError:
        result["has_nvidia_smi"] = False
    
    # 2. PyTorch CUDA
    import torch
    result["torch_version"] = torch.__version__
    result["cuda_available"] = torch.cuda.is_available()
    if result["cuda_available"]:
        result["torch_cuda"] = torch.version.cuda
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["gpu_count"] = torch.cuda.device_count()
        result["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
    
    # 3. bitsandbytes
    try:
        import bitsandbytes
        result["bnb_available"] = True
        result["bnb_version"] = bitsandbytes.__version__
    except ImportError:
        result["bnb_available"] = False
    
    # 4. Recommandation
    if result["cuda_available"]:
        result.update(recommend_method(
            vram_gb=result.get("vram_total_gb", 0),
            model_params_b=0,  # sera calculé avec le modèle choisi
            cuda_available=True,
            bnb_available=result["bnb_available"]
        ))
    
    return result
```

## 4. SFT Engine — Structure

### 4.1 Pipeline de training

```python
class SFTEngine:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def setup_model(self):
        """Charger le modèle avec la bonne quantification."""
        if self.config.method == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=False,
            )
        else:  # lora
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=False,
            )
        
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="auto",
        )
        
        return model, peft_config
    
    def setup_tokenizer(self):
        """Charger et configurer le tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer
    
    def load_dataset(self):
        """Charger train.jsonl et eval.jsonl."""
        # ...
    
    def train(self, callback=None):
        """Lancer le training avec callbacks pour les logs."""
        # ...
```

### 4.2 Callback pour logs temps réel

```python
class StreamingCallback(TrainerCallback):
    """Publie les métriques vers Redis Pub/Sub."""
    
    def __init__(self, redis_client, job_id):
        self.redis = redis_client
        self.channel = f"logs:{job_id}"
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.redis.publish(self.channel, json.dumps({
                "event": "log",
                "step": state.global_step,
                "total_steps": state.max_steps,
                **logs
            }))
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.redis.publish(self.channel, json.dumps({
                "event": "eval",
                "step": state.global_step,
                **metrics
            }))
    
    def on_save(self, args, state, control, **kwargs):
        self.redis.publish(self.channel, json.dumps({
            "event": "checkpoint",
            "step": state.global_step
        }))
    
    def on_train_end(self, args, state, control, **kwargs):
        self.redis.publish(self.channel, json.dumps({
            "event": "complete",
            "total_steps": state.global_step,
            "best_metric": state.best_metric
        }))
```

## 5. Eval Engine — Métriques

### 5.1 Métriques automatiques (MVP)

| Métrique | Source | Description |
|----------|--------|-------------|
| `train_loss` | SFTTrainer | Loss final sur le train set |
| `eval_loss` | SFTTrainer | Loss final sur l'eval set |
| `train_loss_curve` | Logs | Array de {step, loss} pour graphique |
| `eval_loss_curve` | Logs | Array de {step, loss} pour graphique |
| `perplexity` | `exp(eval_loss)` | Perplexité sur eval set |
| `train_runtime` | SFTTrainer | Temps total en secondes |
| `train_samples_per_second` | SFTTrainer | Throughput |

### 5.2 Exemples d'inférence (qualitative)

Après training, générer des réponses sur 5-10 exemples de l'eval set :

```json
{
  "inference_samples": [
    {
      "instruction": "Explique...",
      "input": "...",
      "expected_output": "...",
      "model_output": "...",
      "output_length": 156
    }
  ]
}
```

## 6. Export — Contenu du package

```
export/
  adapter_model.safetensors    # poids LoRA
  adapter_config.json           # config PEFT
  tokenizer.json                # tokenizer (si modifié)
  tokenizer_config.json
  special_tokens_map.json
  training_config.json          # hyperparams utilisés
  metrics.json                  # toutes les métriques
  report.md                     # rapport lisible
  train_loss.png                # graphique loss (optionnel)
  finetuneflow_export.zip       # tout zippé
```

### 6.1 Structure `report.md`

```markdown
# FineTuneFlow Training Report

## Project
- Name: {name}
- Task: {task_type}
- Base Model: {base_model_id}

## Dataset
- Train examples: {n_train}
- Eval examples: {n_eval}
- Avg tokens per example: {avg_tokens}

## Training Configuration
- Method: {method} (LoRA/QLoRA)
- Epochs: {num_epochs}
- Learning rate: {lr}
- LoRA rank: {r}, alpha: {alpha}
- Max sequence length: {max_seq_length}
- GPU: {gpu_name}

## Results
- Final train loss: {train_loss}
- Final eval loss: {eval_loss}
- Perplexity: {perplexity}
- Training time: {duration}

## Sample Outputs
| Instruction | Expected | Model Output |
|------------|----------|-------------|
| ... | ... | ... |
```
