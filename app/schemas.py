from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
from uuid import UUID


class CategoryBase(BaseModel):
    name: str
    path: str

    @field_validator('path', mode='before')
    @classmethod
    def convert_ltree_to_str(cls, v):
        return str(v)


class CategoryResponse(CategoryBase):
    id: int

    class Config:
        from_attributes = True


class ProductBase(BaseModel):
    title: str
    description: Optional[str] = None
    price: Optional[float] = None
    image_url: Optional[str] = None
    category_id: Optional[int] = None


class ProductResponse(ProductBase):
    id: int
    created_at: datetime
    search_distance: Optional[float] = None

    class Config:
        from_attributes = True

class UserInteractionCreate(BaseModel):
    user_id: Optional[UUID] = None
    product_id: int
    interaction_type: str = Field(..., description="view, click, purchase")

class UserInteractionResponse(UserInteractionCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class CategorizationLogCreate(BaseModel):
    product_id: int
    predicted_category_id: Optional[int] = None
    confidence_score: float
    is_reviewed: bool = False
    final_category_id: Optional[int] = None

class CategorizationLogResponse(CategorizationLogCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True