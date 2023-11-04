from pydantic import BaseModel


class ConfiguredBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
