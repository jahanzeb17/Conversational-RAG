from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
     llama = "llama-3.3-70b-versatile",
     deepseek = "deepseek-r1-distill-llama-70b"


class QueryInput(BaseModel):
     question: str
     session_id: str = Field(default=None)
     model: ModelName = Field(default=ModelName.llama)


class QueryResponse(BaseModel):
     answer: str
     session_id: str
     model: ModelName


class DocumentInfo(BaseModel):
     id: int
     filename: str
     upload_timestamp: datetime

    
class DeleteFileRequest(BaseModel):
     file_id: int