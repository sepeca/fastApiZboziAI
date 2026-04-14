from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form
from sqlalchemy.orm import Session
from sqlalchemy import text, or_,func,cast, Text
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from uuid import UUID

from sqlalchemy_utils import Ltree
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
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS ltree;"))
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
    cat_data = category.model_dump()

    if "path" in cat_data:
        cat_data["path"] = Ltree(cat_data["path"])

    db_category = models.Category(**cat_data)
    db.add(db_category)
    db.commit()
    db.refresh(db_category)

    db_category.path = str(db_category.path)
    return db_category

@app.get("/categories/", response_model=List[schemas.CategoryResponse], tags=["Categories"])
def get_categories(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(models.Category).offset(skip).limit(limit).all()

@app.post("/products/", response_model=schemas.ProductResponse, tags=["Goods"])
def create_product(product: schemas.ProductBase, db: Session = Depends(get_db)):
    text_to_encode = f"{product.title} {product.description or ''}"
    text_emb = text_model.encode(text_to_encode)
    final_emb = text_emb / np.linalg.norm(text_emb)

    if product.image_url:
        img = load_image_from_url(product.image_url)
        if img:
            img_emb = img_model.encode(img)
            mixed_emb = (text_emb + img_emb) / 2.0
            final_emb = mixed_emb / np.linalg.norm(mixed_emb)

    vector_list = final_emb.tolist()

    predicted_cat_id = None
    confidence_score = 0.0
    needs_review = False

    if product.category_id is None:
        categories = db.query(models.Category).all()
        best_cat_id = None
        min_distance = 2.0

        for cat in categories:
            centroid = cat.category_embedding

            if centroid is None:
                products_in_cat = db.query(models.Product.embedding).filter(
                    models.Product.category_id == cat.id,
                    models.Product.embedding.isnot(None)
                ).all()
                if products_in_cat:
                    vecs = [np.array(p[0]) for p in products_in_cat]
                    calc_centroid = np.mean(vecs, axis=0)
                    calc_centroid = calc_centroid / np.linalg.norm(calc_centroid)
                    cat.category_embedding = calc_centroid.tolist()
                    db.commit()
                    centroid = cat.category_embedding

            if centroid is not None:
                centroid_np = np.array(centroid)
                dot_product = np.dot(final_emb, centroid_np)
                dist = 1.0 - dot_product

                if dist < min_distance:
                    min_distance = dist
                    best_cat_id = cat.id

        if best_cat_id is not None:
            predicted_cat_id = best_cat_id
            confidence_score = max(0.0, (1.0 - (min_distance / 0.65))) * 100.0

            if confidence_score >= 60.0:
                product.category_id = predicted_cat_id
            else:
                needs_review = True

    db_product = models.Product(**product.model_dump(), embedding=vector_list)
    db.add(db_product)
    db.commit()
    db.refresh(db_product)

    if needs_review and predicted_cat_id is not None:
        log_entry = models.CategorizationLog(
            product_id=db_product.id,
            predicted_category_id=predicted_cat_id,
            confidence_score=confidence_score,
            is_reviewed=False
        )
        db.add(log_entry)
        db.commit()

    return db_product

@app.post("/products/upload/", response_model=schemas.ProductResponse, tags=["Goods"])
def create_product_with_local_file(
        title: str = Form(...),
        price: float = Form(...),
        description: Optional[str] = Form(None),
        category_id: Optional[int] = Form(None),
        file: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    image_bytes = file.file.read()
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="Wrong image format")

    text_to_encode = f"{title} {description or ''}"
    text_emb = text_model.encode(text_to_encode)
    text_emb = text_emb / np.linalg.norm(text_emb)

    img_emb = img_model.encode(img)
    mixed_emb = (text_emb + img_emb) / 2.0
    final_emb = mixed_emb / np.linalg.norm(mixed_emb)

    vector_list = final_emb.tolist()

    predicted_cat_id = None
    confidence_score = 0.0
    needs_review = False

    if category_id is None:
        categories = db.query(models.Category).all()
        best_cat_id = None
        min_distance = 2.0

        for cat in categories:
            centroid = cat.category_embedding
            if centroid is None:
                products_in_cat = db.query(models.Product.embedding).filter(
                    models.Product.category_id == cat.id,
                    models.Product.embedding.isnot(None)
                ).all()
                if products_in_cat:
                    vecs = [np.array(p[0]) for p in products_in_cat]
                    calc_centroid = np.mean(vecs, axis=0)
                    calc_centroid = calc_centroid / np.linalg.norm(calc_centroid)
                    cat.category_embedding = calc_centroid.tolist()
                    db.commit()
                    centroid = cat.category_embedding

            if centroid is not None:
                centroid_np = np.array(centroid)
                dist = 1.0 - np.dot(final_emb, centroid_np)
                if dist < min_distance:
                    min_distance = dist
                    best_cat_id = cat.id

        if best_cat_id is not None:
            predicted_cat_id = best_cat_id
            confidence_score = max(0.0, (1.0 - (min_distance / 0.65))) * 100.0

            if confidence_score >= 60.0:
                category_id = predicted_cat_id
            else:
                needs_review = True

    db_product = models.Product(
        title=title,
        description=description,
        price=price,
        category_id=category_id,
        embedding=vector_list,
        image_url=None
    )
    db.add(db_product)
    db.commit()
    db.refresh(db_product)

    if needs_review and predicted_cat_id is not None:
        log_entry = models.CategorizationLog(
            product_id=db_product.id,
            predicted_category_id=int(predicted_cat_id),
            confidence_score=float(confidence_score),
            is_reviewed=False
        )
        db.add(log_entry)
        db.commit()

    return db_product
@app.get("/search/", response_model=List[schemas.ProductResponse], tags=["Search"])
def search_products(query: str,
                    limit: int = 5,
                    threshold: float = 0.35,
                    user_id: Optional[UUID] = None,
                    category_id: Optional[int] = None,
                    db: Session = Depends(get_db)):
    query_vector = text_model.encode(query)
    query_vector = query_vector / np.linalg.norm(query_vector)

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
            user_pref_vector = user_pref_vector / np.linalg.norm(user_pref_vector)

            query_vector = (query_vector * 0.7) + (user_pref_vector * 0.3)
            query_vector = query_vector / np.linalg.norm(query_vector)

    final_query_list = query_vector.tolist()

    distance_metric = models.Product.embedding.cosine_distance(final_query_list).label("distance")
    query_obj = db.query(models.Product, distance_metric)

    if category_id:
        target_category = db.query(models.Category).filter(models.Category.id == category_id).first()

        if target_category:
            path_str = str(target_category.path)

            query_obj = query_obj.join(models.Category).filter(
                cast(models.Category.path, Text).startswith(path_str)
            )
        else:
            raise HTTPException(status_code=404, detail="Category not found")

    vector_condition = models.Product.embedding.cosine_distance(final_query_list) < threshold
    text_condition = func.unaccent(models.Product.title).ilike(func.unaccent(f"%{query}%"))

    results = query_obj.filter(
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
        cat = db.query(models.Category).filter(models.Category.id == final_category_id).first()
        if cat and cat.category_embedding:
            old_emb = np.array(cat.category_embedding)
            prod_emb = np.array(product.embedding)
            new_emb = (old_emb * 0.9) + (prod_emb * 0.1)
            cat.category_embedding = (new_emb / np.linalg.norm(new_emb)).tolist()

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