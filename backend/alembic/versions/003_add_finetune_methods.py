"""add dora, ia3, prefix, full finetune methods

Revision ID: 003_add_finetune_methods
Revises: 002_add_task_types
Create Date: 2026-02-16 00:00:00.000000

Adds new enum values to finetune_method: dora, ia3, prefix, full.
PostgreSQL enums require ALTER TYPE ... ADD VALUE for each new entry.
"""

from alembic import op

# revision identifiers
revision = "003_add_finetune_methods"
down_revision = "002_add_task_types"
branch_labels = None
depends_on = None

NEW_VALUES = ("dora", "ia3", "prefix", "full")


def upgrade() -> None:
    for value in NEW_VALUES:
        op.execute(f"ALTER TYPE finetune_method ADD VALUE IF NOT EXISTS '{value}'")


def downgrade() -> None:
    # PostgreSQL does not support removing values from an enum type.
    # To fully revert, you would need to recreate the enum.
    # This is left as a no-op because the extra values are harmless.
    pass
