from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text, or_,func
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from uuid import UUID

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
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent;"))
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
        print(f"Error image loading by URL {url}: {e}")
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
def search_products(query: str,
                    limit: int = 5,
                    threshold: float = 0.35,
                    user_id: Optional[UUID] = None,
                    db: Session = Depends(get_db)):
    query_vector = text_model.encode(query)

    if user_id:
        recent_interactions = db.query(models.Product.embedding).join(
            models.UserInteraction
        ).filter(
            models.UserInteraction.user_id == user_id,
            models.UserInteraction.interaction_type == 'click'
        ).order_by(models.UserInteraction.created_at.desc()).limit(10).all()

        if recent_interactions:
            user_vectors = [np.array(r[0]) for r in recent_interactions]
            user_pref_vector = np.mean(user_vectors, axis=0)
            query_vector = (query_vector * 0.7) + (user_pref_vector * 0.3)

    final_query_list = query_vector.tolist()

    distance_metric = models.Product.embedding.cosine_distance(final_query_list).label("distance")
    vector_condition = models.Product.embedding.cosine_distance(final_query_list) < threshold
    text_condition = func.unaccent(models.Product.title).ilike(func.unaccent(f"%{query}%"))

    results = db.query(models.Product, distance_metric).filter(
        or_(vector_condition, text_condition)
    ).all()

    final_response = []
    for product, dist in results:
        prod_data = schemas.ProductResponse.model_validate(product).model_dump()
        final_distance = dist

        if query.lower() in product.title.lower():
            final_distance -= 0.20

        prod_data["search_distance"] = round(final_distance, 3)
        final_response.append(prod_data)

    final_response = sorted(final_response, key=lambda x: x["search_distance"])
    return final_response[:limit]


@app.get("/moderation/logs/", response_model=List[schemas.CategorizationLogResponse], tags=["Moderation"])
def get_unreviewed_logs(limit: int = 50, db: Session = Depends(get_db)):
    return db.query(models.CategorizationLog).filter(models.CategorizationLog.is_reviewed == False).limit(limit).all()

@app.put("/moderation/logs/{log_id}/review", response_model=schemas.CategorizationLogResponse, tags=["Moderation"])
def review_categorization(log_id: int, final_category_id: int, db: Session = Depends(get_db)):
    log = db.query(models.CategorizationLog).filter(models.CategorizationLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Moderation log not found")

    log.final_category_id = final_category_id
    log.is_reviewed = True

    product = db.query(models.Product).filter(models.Product.id == log.product_id).first()
    if product:
        product.category_id = final_category_id

    db.commit()
    db.refresh(log)
    return log
@app.post("/interact/", response_model=schemas.UserInteractionResponse, tags=["Personalization"])
def log_interaction(interaction: schemas.UserInteractionCreate, db: Session = Depends(get_db)):
    db_interaction = models.UserInteraction(**interaction.model_dump(exclude_unset=True))
    db.add(db_interaction)
    db.commit()
    db.refresh(db_interaction)
    return db_interaction