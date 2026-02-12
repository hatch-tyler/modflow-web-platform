"""add temporal and grid spacing fields to projects

Revision ID: 002_temporal_grid
Revises: 001_coord_ref
Create Date: 2026-02-08

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = '002_temporal_grid'
down_revision: Union[str, None] = '001_coord_ref'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('projects', sa.Column('stress_period_data', JSONB(), nullable=True))
    op.add_column('projects', sa.Column('start_date', sa.Date(), nullable=True))
    op.add_column('projects', sa.Column('delr', JSONB(), nullable=True))
    op.add_column('projects', sa.Column('delc', JSONB(), nullable=True))
    op.add_column('projects', sa.Column('time_unit', sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'time_unit')
    op.drop_column('projects', 'delc')
    op.drop_column('projects', 'delr')
    op.drop_column('projects', 'start_date')
    op.drop_column('projects', 'stress_period_data')
