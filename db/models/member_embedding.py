from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    ForeignKey,
    DateTime,
    LargeBinary,
    Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from app.db.base import Base


class MemberEmbedding(Base):
    __tablename__ = "member_embeddings"

    id = Column(BigInteger, primary_key=True, index=True)

    # FK: which member this embedding belongs to
    member_id = Column(
        Integer,
        ForeignKey("members.id", ondelete="CASCADE"),
        nullable=False
    )

    # FK: which camera captured it
    camera_id = Column(
        Integer,
        ForeignKey("cameras.id", ondelete="SET NULL"),
        nullable=True   # must be True
    )

    # Aggregated / representative embeddings (pgvector)
    body_embedding = Column(Vector(512), nullable=True)
    face_embedding = Column(Vector(512), nullable=True)
    back_body_embedding = Column(Vector(512), nullable=True)

    # Raw embeddings (for reprocessing / retraining)
    body_embeddings_raw = Column(LargeBinary, nullable=True)
    face_embeddings_raw = Column(LargeBinary, nullable=True)
    back_body_embeddings_raw = Column(LargeBinary, nullable=True)

    # Last update timestamp
    last_embedding_update_ts = Column(
        DateTime(timezone=True),
        nullable=True,
        server_default=func.now()
    )

    # relationships
    member = relationship("Member", backref="embeddings")
    camera = relationship("Camera", backref="member_embeddings")

    __table_args__ = (
        # fast lookup per member + camera
        Index("ix_member_embedding_member_camera", "member_id", "camera_id"),
    )