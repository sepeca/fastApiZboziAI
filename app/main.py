from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO

from . import models, schemas
from .database import engine, get_db

app = FastAPI(title="ZboziAI")

print("Loading text model CLIP (multilingual)...")
text_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')

print("Loading visual model CLIP (image)...")
img_model = SentenceTransformer('clip-ViT-B-32')

print("Both models successfully loaded")

@app.on_event("startup")
def on_startup():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
    models.Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"message": "Welcome to ZboziAI Hybrid Search API"}

def load_image_from_url(url: str):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Ошибка загрузки картинки по URL {url}: {e}")
        return None

@app.post("/categories/", response_model=schemas.CategoryResponse, tags=["Categories"])
def create_category(category: schemas.CategoryBase, db: Session = Depends(get_db)):
    db_category = models.Category(**category.model_dump())
    db.add(db_category)
    db.commit()
    db.refresh(db_category)
    return db_category

@app.get("/categories/", response_model=List[schemas.CategoryResponse], tags=["Categories"])
def get_categories(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(models.Category).offset(skip).limit(limit).all()

@app.post("/products/", response_model=schemas.ProductResponse, tags=["Goods"])
def create_product(product: schemas.ProductBase, db: Session = Depends(get_db)):
    text_to_encode = f"{product.title} {product.description or ''}"
    text_emb = text_model.encode(text_to_encode)

    final_emb = text_emb

    if product.image_url:
        img = load_image_from_url(product.image_url)
        if img:
            img_emb = img_model.encode(img)
            final_emb = (text_emb + img_emb) / 2.0

    vector_list = final_emb.tolist()

    db_product = models.Product(**product.model_dump(), embedding=vector_list)
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product

@app.get("/search/", response_model=List[schemas.ProductResponse], tags=["Search"])
def search_products(query: str, limit: int = 5, threshold: float = 0.7, db: Session = Depends(get_db)):
    query_vector = text_model.encode(query).tolist()

    results = db.query(models.Product).filter(
        models.Product.embedding.cosine_distance(query_vector) < threshold
    ).order_by(
        models.Product.embedding.cosine_distance(query_vector)
    ).limit(limit).all()

    return results