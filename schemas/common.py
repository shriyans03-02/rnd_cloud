from pydantic import BaseModel
from typing import Generic, Optional, TypeVar,List

T = TypeVar("T")

class MessageResponse(BaseModel, Generic[T]):
    message: str
    data: Optional[T] = None

class BulkImportResponse(BaseModel):
    message: str
    added_count: int
    skipped_count: int
    skipped: List[dict]     
