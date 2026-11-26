from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

FunctionalStatus = Literal["working", "untested", "for_parts"]


class VisionResult(BaseModel):
    """What the vision model read off the photo. Empty string = not readable."""

    part_name: str = ""
    brand: str = ""
    part_number: str = ""
    part_number_confidence: float = Field(0.0, ge=0, le=1)
    legibility: Literal["clear", "partial", "blurry", "none"] = "none"
    model_numbers: list[str] = []
    appliance_type: str = ""
    visual_condition: str = ""
    weight_lb_estimate: Optional[float] = None


class NoteFacts(BaseModel):
    """Facts extracted from the technician's spoken/typed note."""

    functional_status: FunctionalStatus = "untested"
    model_numbers: list[str] = []
    condition_notes: str = ""


class Comp(BaseModel):
    title: str = ""
    price: float
    shipping: float = 0.0
    sold: bool = False


class PricePoints(BaseModel):
    low: float = 0.0   # quick sale
    avg: float = 0.0   # market average
    high: float = 0.0  # refurbished / premium


class PricingInfo(BaseModel):
    estimated_price_usd: PricePoints = PricePoints()
    sample_size: int = 0
    confidence: Literal["none", "low", "medium", "high"] = "none"
    source: str = ""
    query: str = ""


class AnalysisResult(BaseModel):
    """Public output. The first block matches the spec'd listing JSON."""

    listing_title: str
    brand: str
    part_number: str
    compatible_models: list[str]
    condition: str
    suggested_description: str
    estimated_price_usd: PricePoints
    shipping_weight_estimate: str
    # --- extras the technician needs to trust the result ---
    needs_review: bool
    flags: list[str] = []
    price_confidence: str = "none"
    price_sample_size: int = 0
    functional_status: FunctionalStatus = "untested"
    transcript: str = ""
    timings_ms: dict[str, int] = {}
    demo: bool = False
    currency: str = "USD"
    id: Optional[int] = None          # saved item id (for corrections/export)
    confirmed: bool = False           # a human checked/edited this item
