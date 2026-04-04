from sqlalchemy import Column, Integer, String, Text, Numeric, ForeignKey, DateTime, Computed, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import TSVECTOR
from pgvector.sqlalchemy import Vector
from .database import Base


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    parent_id = Column(Integer, ForeignKey("categories.id"), nullable=True)

    category_embedding = Column(Vector(512), nullable=True)

    products = relationship("Product", back_populates="category")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String(100), unique=True, nullable=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    price = Column(Numeric(12, 2), nullable=True)
    image_url = Column(String(500), nullable=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True)

    embedding = Column(Vector(512), nullable=True)

    fts_vector = Column(
        TSVECTOR,
        Computed("to_tsvector('simple', title || ' ' || coalesce(description, ''))", persisted=True)
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    category = relationship("Category", back_populates="products")

    __table_args__ = (
        Index('product_hnsw_idx', 'embedding', postgresql_using='hnsw',
              postgresql_with={'m': 16, 'ef_construction': 64}, postgresql_ops={'embedding': 'vector_cosine_ops'}),
        Index('product_fts_idx', 'fts_vector', postgresql_using='gin'),
    )