"""add grid_type column to projects

Revision ID: 003_grid_type
Revises: 002_temporal_grid
Create Date: 2026-02-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_grid_type'
down_revision: Union[str, None] = '002_temporal_grid'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('projects', sa.Column('grid_type', sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'grid_type')
