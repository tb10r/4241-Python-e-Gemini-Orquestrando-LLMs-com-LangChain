from pydantic import BaseModel, Field
from typing import List

class DetalhesImagemModelo(BaseModel):
    titulo: str = Field(
        description="Defina o titulo adequado para a imagem que foi analisada."
        )
    
    descricao: str = Field(
        description="coloque aqui uma descrição detalhada da sua análise para imagem."
        )
    
    rotulos: List[str] = Field(
        description="defina três rótulos para a imagem analisada."
        )