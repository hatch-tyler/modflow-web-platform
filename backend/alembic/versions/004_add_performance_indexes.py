"""add performance indexes to runs and projects tables

Revision ID: 004_perf_indexes
Revises: 003_grid_type
Create Date: 2026-02-12

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '004_perf_indexes'
down_revision: Union[str, None] = '003_grid_type'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Single-column indexes on runs
    op.create_index('ix_runs_project_id', 'runs', ['project_id'])
    op.create_index('ix_runs_status', 'runs', ['status'])
    op.create_index('ix_runs_run_type', 'runs', ['run_type'])
    op.create_index('ix_runs_created_at', 'runs', ['created_at'])

    # Composite indexes on runs (common query patterns)
    op.create_index('idx_runs_project_status', 'runs', ['project_id', 'status'])
    op.create_index('idx_runs_project_created', 'runs', ['project_id', 'created_at'])

    # Timestamp indexes on projects
    op.create_index('ix_projects_created_at', 'projects', ['created_at'])
    op.create_index('ix_projects_updated_at', 'projects', ['updated_at'])


def downgrade() -> None:
    op.drop_index('ix_projects_updated_at', table_name='projects')
    op.drop_index('ix_projects_created_at', table_name='projects')
    op.drop_index('idx_runs_project_created', table_name='runs')
    op.drop_index('idx_runs_project_status', table_name='runs')
    op.drop_index('ix_runs_created_at', table_name='runs')
    op.drop_index('ix_runs_run_type', table_name='runs')
    op.drop_index('ix_runs_status', table_name='runs')
    op.drop_index('ix_runs_project_id', table_name='runs')
