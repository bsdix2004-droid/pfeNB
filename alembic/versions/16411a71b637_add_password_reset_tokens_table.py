"""add password reset tokens table

Revision ID: 16411a71b637
Revises: a844dae5eb76
Create Date: 2026-04-29 01:34:36.327347

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '16411a71b637'
down_revision: Union[str, Sequence[str], None] = 'a844dae5eb76'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create password reset tokens table
    op.create_table('password_reset_tokens',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('token_hash', sa.String(length=255), nullable=False),
    sa.Column('is_used', sa.Boolean(), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_password_reset_tokens_token_hash'), 'password_reset_tokens', ['token_hash'], unique=True)
    op.create_index(op.f('ix_password_reset_tokens_user_id'), 'password_reset_tokens', ['user_id'], unique=False)
    
    # Remove username column from users
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_column('users', 'username')
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # Restore username column
    op.add_column('users', sa.Column('username', sa.VARCHAR(length=50), autoincrement=False, nullable=False))
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    
    # Drop password reset tokens table
    op.drop_index(op.f('ix_password_reset_tokens_user_id'), table_name='password_reset_tokens')
    op.drop_index(op.f('ix_password_reset_tokens_token_hash'), table_name='password_reset_tokens')
    op.drop_table('password_reset_tokens')