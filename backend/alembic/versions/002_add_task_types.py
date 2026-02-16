"""add 7 task types

Revision ID: 002_add_task_types
Revises: 001_initial
Create Date: 2026-02-15 00:00:00.000000

Migrates the task_type enum from (instruction, qa) to 7 new task types.
Existing rows are remapped: instruction → instruction_tuning, qa → qa_grounded.
Old enum values are fully removed by recreating the PostgreSQL enum type.
"""

from alembic import op
from sqlalchemy import text

# revision identifiers
revision = "002_add_task_types"
down_revision = "001_initial"
branch_labels = None
depends_on = None

NEW_VALUES = (
    "instruction_tuning",
    "qa_grounded",
    "summarization",
    "report_generation",
    "information_extraction",
    "classification",
    "chat_dialogue_sft",
)

OLD_VALUES = ("instruction", "qa")


def upgrade() -> None:
    conn = op.get_bind()

    # ── Step 1: Add new enum values (requires own committed transaction) ──
    conn.execute(text("COMMIT"))
    for val in NEW_VALUES:
        conn.execute(text(f"ALTER TYPE task_type ADD VALUE IF NOT EXISTS '{val}'"))
    conn.execute(text("BEGIN"))

    # ── Step 2: Remap data from old → new ──
    conn.execute(text(
        "UPDATE projects SET task_type = 'instruction_tuning' WHERE task_type = 'instruction'"
    ))
    conn.execute(text(
        "UPDATE projects SET task_type = 'qa_grounded' WHERE task_type = 'qa'"
    ))

    # ── Step 3: Recreate enum WITHOUT old values ──
    # PostgreSQL has no ALTER TYPE DROP VALUE, so we rename → create → cast → drop.
    # Must also drop/reset the column DEFAULT which references the old type.
    conn.execute(text(
        "ALTER TABLE projects ALTER COLUMN task_type DROP DEFAULT"
    ))
    conn.execute(text("ALTER TYPE task_type RENAME TO task_type_old"))
    conn.execute(text(
        "CREATE TYPE task_type AS ENUM ("
        + ", ".join(f"'{v}'" for v in NEW_VALUES)
        + ")"
    ))
    conn.execute(text(
        "ALTER TABLE projects "
        "ALTER COLUMN task_type TYPE task_type USING task_type::text::task_type"
    ))
    conn.execute(text(
        "ALTER TABLE projects "
        "ALTER COLUMN task_type SET DEFAULT 'instruction_tuning'::task_type"
    ))
    conn.execute(text("DROP TYPE task_type_old"))


def downgrade() -> None:
    conn = op.get_bind()

    # Drop default, remap data, recreate old enum
    conn.execute(text(
        "ALTER TABLE projects ALTER COLUMN task_type DROP DEFAULT"
    ))
    conn.execute(text(
        "UPDATE projects SET task_type = 'instruction_tuning' WHERE task_type NOT IN ('instruction_tuning','qa_grounded')"
    ))
    conn.execute(text(
        "UPDATE projects SET task_type = 'instruction' WHERE task_type = 'instruction_tuning'"
    ))
    conn.execute(text(
        "UPDATE projects SET task_type = 'qa' WHERE task_type = 'qa_grounded'"
    ))

    conn.execute(text("ALTER TYPE task_type RENAME TO task_type_new"))
    conn.execute(text("CREATE TYPE task_type AS ENUM ('instruction', 'qa')"))

    # Cast column back
    conn.execute(text(
        "ALTER TABLE projects "
        "ALTER COLUMN task_type TYPE task_type USING task_type::text::task_type"
    ))
    conn.execute(text(
        "ALTER TABLE projects "
        "ALTER COLUMN task_type SET DEFAULT 'instruction'::task_type"
    ))
    conn.execute(text("DROP TYPE task_type_new"))
