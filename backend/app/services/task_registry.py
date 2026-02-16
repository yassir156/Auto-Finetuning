"""
FineTuneFlow — Task Type Registry.

Centralised configuration for every supported fine-tuning task type.
Each entry specifies:
  - Prompt template for the LLM dataset generator
  - Required / optional fields & validation rules
  - Repair schema for the LLM repair pipeline
  - SFT conversion function (native → instruction/input/output)
  - Frontend display metadata (columns, sample example)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# ════════════════════════════════════════════════════════════
#  Data-class for one Task Type configuration
# ════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TaskConfig:
    """Immutable configuration for a single task type."""

    key: str                      # e.g. "instruction"
    label: str                    # Human-readable label
    description: str              # Short description for UI
    required_fields: list[str]    # Fields that MUST be present & non-empty
    optional_fields: list[str]    # May be present, defaults provided
    prompt_template: str          # Prompt template with {N} and {chunk_text}
    repair_schema: str            # JSON schema string for the LLM repair prompt
    sample_example: dict          # Sample JSONL object shown in frontend
    display_columns: list[dict]   # Column defs [{key, label, maxWidth?}]
    # Validation constraints
    min_output_len: int = 10      # Minimum chars for main output field
    max_field_len: int = 10000    # Max chars per string field
    # Which field is the "main output" for length validation
    main_output_field: str = "output"


# ════════════════════════════════════════════════════════════
#  Prompt Templates
# ════════════════════════════════════════════════════════════

INSTRUCTION_TUNING_TEMPLATE = """TASK: Create {N} training examples for instruction-tuning from the provided source text.

SCHEMA (each line is a JSON object):
{{"instruction": "a clear user request or question", "input": "additional context if needed, or empty string", "output": "a helpful answer grounded in the source text"}}

RULES:
1. Use ONLY facts present in the source text. No hallucinations.
2. "instruction" should be a natural user request (vary phrasing: explain, describe, list, compare, summarize, etc.).
3. "input" may be an empty string "" if the instruction is self-contained.
4. "output" must be a helpful, accurate answer fully supported by the source.
5. Vary difficulty: some simple factual, some requiring synthesis.
6. Vary style: some formal, some conversational.
7. Each example must be on its own line as a valid JSON object.
8. Output EXACTLY {N} lines. No more, no less.
9. Do NOT wrap output in markdown code fences.

SOURCE TEXT:
<<<
{chunk_text}
>>>

Output {N} JSONL lines now:"""

QA_GROUNDED_TEMPLATE = """TASK: Generate {N} question-answer pairs grounded in the provided source text.

SCHEMA (each line is a JSON object):
{{"question": "a clear question", "context": "relevant excerpt from source", "answer": "accurate answer based on context"}}

RULES:
1. The answer MUST be fully supported by the context.
2. "context" must be a relevant excerpt from the source text (not the whole source).
3. Questions should vary in type: factual (who/what/when), analytical (why/how), comparative, inferential.
4. Questions should vary in difficulty: easy, medium, hard.
5. Answers should be concise but complete (2-5 sentences typically).
6. Each example must be on its own line as a valid JSON object.
7. Output EXACTLY {N} lines.
8. Do NOT wrap output in markdown code fences.

SOURCE TEXT:
<<<
{chunk_text}
>>>

Output {N} JSONL lines now:"""

SUMMARIZATION_TEMPLATE = """TASK: Generate {N} summarization training examples from the provided source text.

SCHEMA (each line is a JSON object):
{{"text": "a passage from the source", "summary": "a concise summary of that passage"}}

RULES:
1. "text" should be a substantial paragraph or multi-paragraph passage from the source.
2. "summary" should faithfully condense the key points (2-4 sentences typically).
3. Vary the length of passages: some short (1 paragraph), some long (3-4 paragraphs).
4. Do NOT add information not present in the source.
5. Each example must be on its own line as a valid JSON object.
6. Output EXACTLY {N} lines.
7. Do NOT wrap output in markdown code fences.

SOURCE TEXT:
<<<
{chunk_text}
>>>

Output {N} JSONL lines now:"""

REPORT_GENERATION_TEMPLATE = """TASK: Generate {N} report-generation training examples from the provided source text.

SCHEMA (each line is a JSON object):
{{"input_material": "raw notes or data points extracted from the source", "report": {{"title": "report title", "sections": ["section1 text", "section2 text"], "conclusion": "concluding summary"}}}}

RULES:
1. "input_material" should be raw notes, bullet points, or data extracted from the source.
2. "report" must be a structured JSON object with title, sections array, and conclusion.
3. The report should faithfully organize and expand on the input material.
4. Vary complexity: some simple 2-section reports, some more detailed.
5. Each example must be on its own line as a valid JSON object.
6. Output EXACTLY {N} lines.
7. Do NOT wrap output in markdown code fences.

SOURCE TEXT:
<<<
{chunk_text}
>>>

Output {N} JSONL lines now:"""

INFORMATION_EXTRACTION_TEMPLATE = """TASK: Generate {N} information-extraction training examples from the provided source text.

SCHEMA (each line is a JSON object):
{{"text": "a passage from the source", "extracted": {{"entities": ["entity1", "entity2"], "relations": ["entity1 relation entity2"], "key_facts": ["fact1", "fact2"]}}}}

RULES:
1. "text" should be a passage from the source containing extractable information.
2. "extracted" must contain: entities (named entities), relations (between entities), key_facts (important facts).
3. Only extract information actually present in the text passage.
4. Vary the complexity: some passages with few entities, some with many.
5. Each example must be on its own line as a valid JSON object.
6. Output EXACTLY {N} lines.
7. Do NOT wrap output in markdown code fences.

SOURCE TEXT:
<<<
{chunk_text}
>>>

Output {N} JSONL lines now:"""

CLASSIFICATION_TEMPLATE = """TASK: Generate {N} text classification training examples from the provided source text.

SCHEMA (each line is a JSON object):
{{"text": "a passage or sentence from the source", "label": "appropriate category label"}}

RULES:
1. "text" should be a passage or sentence from the source text.
2. "label" should be a relevant category/topic label that accurately describes the text.
3. Use consistent label names across examples (e.g. "technical", "opinion", "factual", "procedural", etc.).
4. Vary text lengths: some short sentences, some longer paragraphs.
5. Each example must be on its own line as a valid JSON object.
6. Output EXACTLY {N} lines.
7. Do NOT wrap output in markdown code fences.

SOURCE TEXT:
<<<
{chunk_text}
>>>

Output {N} JSONL lines now:"""

CHAT_DIALOGUE_TEMPLATE = """TASK: Generate {N} multi-turn chat dialogue training examples from the provided source text.

SCHEMA (each line is a JSON object):
{{"messages": [{{"role": "user", "content": "user question or request"}}, {{"role": "assistant", "content": "helpful response based on source"}}]}}

RULES:
1. Each example must have a "messages" array with alternating user/assistant turns.
2. Minimum 2 messages (1 user + 1 assistant), maximum 6 messages (3 turns).
3. All assistant responses must be grounded in the source text.
4. User messages should be natural conversational requests.
5. Vary the number of turns across examples.
6. Each example must be on its own line as a valid JSON object.
7. Output EXACTLY {N} lines.
8. Do NOT wrap output in markdown code fences.

SOURCE TEXT:
<<<
{chunk_text}
>>>

Output {N} JSONL lines now:"""


# ════════════════════════════════════════════════════════════
#  SFT Conversion Functions
# ════════════════════════════════════════════════════════════

def _sft_instruction(data: dict) -> dict:
    """instruction_tuning → already in SFT format."""
    return {
        "instruction": data.get("instruction", ""),
        "input": data.get("input", ""),
        "output": data.get("output", ""),
    }


def _sft_qa(data: dict) -> dict:
    """qa_grounded → SFT format."""
    return {
        "instruction": f"Answer the following question using the provided context.\n\nContext: {data.get('context', '')}\n\nQuestion: {data.get('question', '')}",
        "input": "",
        "output": data.get("answer", ""),
    }


def _sft_summarization(data: dict) -> dict:
    """summarization → SFT format."""
    return {
        "instruction": "Summarize the following text concisely.",
        "input": data.get("text", ""),
        "output": data.get("summary", ""),
    }


def _sft_report(data: dict) -> dict:
    """report_generation → SFT format."""
    import json
    report = data.get("report", {})
    if isinstance(report, str):
        report_str = report
    else:
        report_str = json.dumps(report, ensure_ascii=False)
    return {
        "instruction": "Generate a structured report from the following material.",
        "input": data.get("input_material", ""),
        "output": report_str,
    }


def _sft_extraction(data: dict) -> dict:
    """information_extraction → SFT format."""
    import json
    extracted = data.get("extracted", {})
    if isinstance(extracted, str):
        extracted_str = extracted
    else:
        extracted_str = json.dumps(extracted, ensure_ascii=False)
    return {
        "instruction": "Extract entities, relations, and key facts from the following text.",
        "input": data.get("text", ""),
        "output": extracted_str,
    }


def _sft_classification(data: dict) -> dict:
    """classification → SFT format."""
    return {
        "instruction": "Classify the following text into an appropriate category.",
        "input": data.get("text", ""),
        "output": data.get("label", ""),
    }


def _sft_chat(data: dict) -> dict:
    """chat_dialogue_sft → SFT format (last assistant turn as output)."""
    messages = data.get("messages", [])
    if not messages:
        return {"instruction": "", "input": "", "output": ""}

    # Build instruction from all messages except the last assistant message
    parts = []
    last_assistant = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            last_assistant = content
        parts.append(f"{role}: {content}")

    # The instruction is the conversation context, output is the last assistant response
    if len(messages) >= 2:
        context_msgs = messages[:-1]  # All but last
        instruction = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in context_msgs)
        return {
            "instruction": instruction,
            "input": "",
            "output": last_assistant,
        }
    return {"instruction": parts[0] if parts else "", "input": "", "output": last_assistant}


# ════════════════════════════════════════════════════════════
#  Registry — map of task_type → TaskConfig
# ════════════════════════════════════════════════════════════

TASK_CONFIGS: dict[str, TaskConfig] = {
    "instruction_tuning": TaskConfig(
        key="instruction_tuning",
        label="Instruction Tuning",
        description="The model learns to follow instructions and generate helpful responses.",
        required_fields=["instruction", "output"],
        optional_fields=["input"],
        prompt_template=INSTRUCTION_TUNING_TEMPLATE,
        repair_schema='{"instruction": "string", "input": "string", "output": "string"}',
        sample_example={
            "instruction": "Explain how photosynthesis works.",
            "input": "",
            "output": "Photosynthesis is the process by which plants convert sunlight into energy..."
        },
        display_columns=[
            {"key": "instruction", "label": "Instruction", "maxWidth": 300},
            {"key": "input", "label": "Input", "maxWidth": 200},
            {"key": "output", "label": "Output", "maxWidth": 300},
        ],
        main_output_field="output",
    ),
    "qa_grounded": TaskConfig(
        key="qa_grounded",
        label="Q&A (Grounded)",
        description="The model learns to answer questions based on provided context.",
        required_fields=["question", "context", "answer"],
        optional_fields=[],
        prompt_template=QA_GROUNDED_TEMPLATE,
        repair_schema='{"question": "string", "context": "string", "answer": "string"}',
        sample_example={
            "question": "What is the capital of France?",
            "context": "France is a country in Europe. Its capital city is Paris.",
            "answer": "The capital of France is Paris."
        },
        display_columns=[
            {"key": "question", "label": "Question", "maxWidth": 250},
            {"key": "context", "label": "Context", "maxWidth": 250},
            {"key": "answer", "label": "Answer", "maxWidth": 300},
        ],
        main_output_field="answer",
    ),
    "summarization": TaskConfig(
        key="summarization",
        label="Summarization",
        description="The model learns to produce concise summaries of text passages.",
        required_fields=["text", "summary"],
        optional_fields=[],
        prompt_template=SUMMARIZATION_TEMPLATE,
        repair_schema='{"text": "string", "summary": "string"}',
        sample_example={
            "text": "The Industrial Revolution was a period of major industrialization...",
            "summary": "The Industrial Revolution transformed manufacturing through mechanization."
        },
        display_columns=[
            {"key": "text", "label": "Text", "maxWidth": 350},
            {"key": "summary", "label": "Summary", "maxWidth": 350},
        ],
        main_output_field="summary",
    ),
    "report_generation": TaskConfig(
        key="report_generation",
        label="Report Generation",
        description="The model learns to generate structured reports from raw material.",
        required_fields=["input_material", "report"],
        optional_fields=[],
        prompt_template=REPORT_GENERATION_TEMPLATE,
        repair_schema='{"input_material": "string", "report": {"title": "string", "sections": ["string"], "conclusion": "string"}}',
        sample_example={
            "input_material": "- Revenue Q1: $2M\n- Revenue Q2: $2.5M\n- Headcount: +15",
            "report": {
                "title": "Q2 Performance Report",
                "sections": ["Revenue grew 25% QoQ from $2M to $2.5M.", "Team expanded by 15 new hires."],
                "conclusion": "Strong growth trajectory in both revenue and team size."
            }
        },
        display_columns=[
            {"key": "input_material", "label": "Input Material", "maxWidth": 300},
            {"key": "report", "label": "Report", "maxWidth": 400},
        ],
        main_output_field="report",
        min_output_len=5,
    ),
    "information_extraction": TaskConfig(
        key="information_extraction",
        label="Information Extraction",
        description="The model learns to extract entities, relations, and key facts from text.",
        required_fields=["text", "extracted"],
        optional_fields=[],
        prompt_template=INFORMATION_EXTRACTION_TEMPLATE,
        repair_schema='{"text": "string", "extracted": {"entities": ["string"], "relations": ["string"], "key_facts": ["string"]}}',
        sample_example={
            "text": "Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976.",
            "extracted": {
                "entities": ["Apple Inc.", "Steve Jobs", "Steve Wozniak"],
                "relations": ["Steve Jobs founded Apple Inc.", "Steve Wozniak founded Apple Inc."],
                "key_facts": ["Apple Inc. was founded in 1976"]
            }
        },
        display_columns=[
            {"key": "text", "label": "Text", "maxWidth": 350},
            {"key": "extracted", "label": "Extracted", "maxWidth": 350},
        ],
        main_output_field="extracted",
        min_output_len=5,
    ),
    "classification": TaskConfig(
        key="classification",
        label="Classification",
        description="The model learns to categorize text into predefined labels.",
        required_fields=["text", "label"],
        optional_fields=[],
        prompt_template=CLASSIFICATION_TEMPLATE,
        repair_schema='{"text": "string", "label": "string"}',
        sample_example={
            "text": "The patient presented with a fever of 39°C and persistent cough.",
            "label": "medical_symptom"
        },
        display_columns=[
            {"key": "text", "label": "Text", "maxWidth": 400},
            {"key": "label", "label": "Label", "maxWidth": 200},
        ],
        main_output_field="label",
        min_output_len=1,
    ),
    "chat_dialogue_sft": TaskConfig(
        key="chat_dialogue_sft",
        label="Chat / Dialogue SFT",
        description="The model learns multi-turn conversational patterns.",
        required_fields=["messages"],
        optional_fields=[],
        prompt_template=CHAT_DIALOGUE_TEMPLATE,
        repair_schema='{"messages": [{"role": "user|assistant", "content": "string"}]}',
        sample_example={
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI that enables systems to learn from data."}
            ]
        },
        display_columns=[
            {"key": "messages", "label": "Messages", "maxWidth": 600},
        ],
        main_output_field="messages",
        min_output_len=1,
    ),
}

# Backwards compatibility aliases (old key → new key)
_ALIASES: dict[str, str] = {
    "instruction": "instruction_tuning",
    "qa": "qa_grounded",
}


def get_task_config(task_type: str) -> TaskConfig:
    """Get the TaskConfig for a given task type key (supports aliases)."""
    resolved = _ALIASES.get(task_type, task_type)
    if resolved not in TASK_CONFIGS:
        raise ValueError(f"Unknown task type: {task_type}")
    return TASK_CONFIGS[resolved]


def resolve_task_key(task_type: str) -> str:
    """Resolve aliases to canonical key."""
    return _ALIASES.get(task_type, task_type)


ALL_TASK_KEYS = list(TASK_CONFIGS.keys())

SFT_CONVERTERS: dict[str, Callable[[dict], dict]] = {
    "instruction_tuning": _sft_instruction,
    "qa_grounded": _sft_qa,
    "summarization": _sft_summarization,
    "report_generation": _sft_report,
    "information_extraction": _sft_extraction,
    "classification": _sft_classification,
    "chat_dialogue_sft": _sft_chat,
}


def to_sft_format(task_type: str, data: dict) -> dict:
    """Convert native task data to SFT instruction/input/output format."""
    key = resolve_task_key(task_type)
    converter = SFT_CONVERTERS.get(key, _sft_instruction)
    return converter(data)
