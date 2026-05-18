# ZboziAI - Inteligentní backend pro E-commerce VE VÝVOJI ([ENG Version](https://github.com/sepeca/fastApiZboziAI/blob/main/README_ENG.md))

ZboziAI je API postavené na frameworku FastAPI, které poskytuje možnosti hybridního vyhledávání, multimediálního zpracování dat (text a obrázky), automatické kategorizace zboží a personalizace výsledků pro uživatele. Jako úložiště využívá PostgreSQL s rozšířeními pro práci s vektory, plnotextovým vyhledáváním a hierarchickými strukturami.

## Hlavní funkce

* **Multimodální vektory (Embeddings):** Využití modelů CLIP (`clip-ViT-B-32-multilingual-v1` pro text a `clip-ViT-B-32` pro obrázky) k vytvoření jednotné sémantické reprezentace produktu.
* **Hybridní vyhledávání (Hybrid Search):** Spojení sémantického vektorového vyhledávání (pgvector), neexaktního vyhledávání pomocí trigramů (pg_trgm) a plnotextového vyhledávání (FTS) s využitím algoritmů hodnocení RRF (Reciprocal Rank Fusion).
* **Automatická kategorizace:** Určení kategorie nového produktu za běhu pomocí výpočtu kosinové vzdálenosti mezi vektorem produktu a centroidy existujících kategorií.
* **Systém moderování a Active Learning:** Produkty s nízkou jistotou klasifikace jsou odesílány k moderování. Při ručním potvrzení se centroid kategorie přepočítá, což zlepšuje přesnost budoucích predikcí.
* **Personalizace:** Jemný posun vektoru vyhledávacího dotazu uživatele směrem k jeho předchozím interakcím (kliknutí/zobrazení) pro poskytování personalizovaných výsledků.
* **Hierarchie kategorií:** Využití rozšíření PostgreSQL `ltree` pro striktní a rychlé filtrování výsledků podle stromu kategorií.

## Technologický stoh

* **Framework:** FastAPI, Pydantic
* **Databáze:** PostgreSQL 15+
* **Rozšíření databáze:** `vector` (pgvector), `pg_trgm`, `unaccent`, `ltree`
* **ORM:** SQLAlchemy
* **Strojové učení:** Sentence-Transformers (PyTorch), NumPy, Pillow (PIL)
