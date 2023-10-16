from typing import Tuple

from numpy._typing import ArrayLike
from pydantic import BaseModel, PositiveFloat, PositiveInt


class PSFAnalysisInputs(BaseModel):
    microscope: str
    magnification: PositiveFloat
    na: PositiveFloat
    spacing: Tuple[PositiveFloat, PositiveFloat, PositiveFloat]
    patch_size: Tuple[PositiveInt, PositiveInt, PositiveInt]
    name: str
    img_data: ArrayLike
    point_data: ArrayLike
    dpi: PositiveInt
    date: str
    version: str

    class Config:
        arbitrary_types_allowed = True
