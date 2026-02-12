"""add coordinate reference fields to projects

Revision ID: 001_coord_ref
Revises: None
Create Date: 2026-02-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_coord_ref'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('projects', sa.Column('xoff', sa.Float(), nullable=True))
    op.add_column('projects', sa.Column('yoff', sa.Float(), nullable=True))
    op.add_column('projects', sa.Column('angrot', sa.Float(), nullable=True))
    op.add_column('projects', sa.Column('epsg', sa.Integer(), nullable=True))
    op.add_column('projects', sa.Column('length_unit', sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'length_unit')
    op.drop_column('projects', 'epsg')
    op.drop_column('projects', 'angrot')
    op.drop_column('projects', 'yoff')
    op.drop_column('projects', 'xoff')
