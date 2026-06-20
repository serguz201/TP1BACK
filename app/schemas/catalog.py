import uuid
from pydantic import BaseModel


class PortCatalogResponse(BaseModel):
    key: str
    name: str


class ContainerTypeResponse(BaseModel):
    id: uuid.UUID
    code: str
    name: str
    volume_cbm: float
    max_weight_kg: float

    class Config:
        from_attributes = True
