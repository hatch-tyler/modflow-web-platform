"""add completed_at index to runs table

Revision ID: 005_completed_at_idx
Revises: 004_perf_indexes
Create Date: 2026-03-11

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '005_completed_at_idx'
down_revision: Union[str, None] = '004_perf_indexes'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index('idx_runs_completed_at', 'runs', ['completed_at'])


def downgrade() -> None:
    op.drop_index('idx_runs_completed_at', table_name='runs')
