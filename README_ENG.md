# ZboziAI - Intelligent Backend for E-commerce IN PROGRESS
ZboziAI is a modern API built on the FastAPI framework that provides hybrid search capabilities, multimodal data processing (text and images), automatic product categorization, and user result personalization. It uses PostgreSQL as storage, equipped with extensions for vector operations, full-text search, and hierarchical structures.

## Core Features

* **Multimodal Vectors (Embeddings):** Utilizing CLIP models (`clip-ViT-B-32-multilingual-v1` for text and `clip-ViT-B-32` for images) to create a unified semantic representation of a product.
* **Hybrid Search:** Combining semantic vector search (pgvector), fuzzy text matching using trigrams (pg_trgm), and Full-Text Search (FTS) using the Reciprocal Rank Fusion (RRF) ranking algorithm.
* **Automatic Categorization:** Determining the category of a new product on the fly by calculating the cosine distance between the product vector and the centroids of existing categories.
* **Moderation System and Active Learning:** Products with low classification confidence are sent for moderation. Upon manual confirmation, the category centroid is recalculated, improving the accuracy of future predictions.
* **Personalization:** Slightly shifting the user's search query vector towards their previous interactions (clicks/views) to deliver personalized results.
* **Category Hierarchy:** Utilizing the PostgreSQL `ltree` extension for strict and fast result filtering across the category tree.

## Tech Stack

* **Framework:** FastAPI, Pydantic
* **Database:** PostgreSQL 15+
* **Database Extensions:** `vector` (pgvector), `pg_trgm`, `unaccent`, `ltree`
* **ORM:** SQLAlchemy
* **Machine Learning:** Sentence-Transformers (PyTorch), NumPy, Pillow (PIL)
