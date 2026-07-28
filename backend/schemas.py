from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import uuid
import time

class SliderBounds(BaseModel):
    min: Optional[str] = "-10"
    max: Optional[str] = "10"
    step: Optional[str] = "0.1"

class DesmosExpression(BaseModel):
    id: str = Field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}", description="Unique expression ID for Desmos API")
    latex: str = Field(..., description="Valid LaTeX string compatible with Desmos graphing engine")
    color: Optional[str] = Field(default="#2d70b3", description="Color hex code or name for rendering")
    lineStyle: Optional[str] = Field(default="SOLID", description="SOLID, DASHED, or DOTTED")
    lineWidth: Optional[float] = Field(default=2.5, description="Line width in pixels")
    hidden: Optional[bool] = Field(default=False, description="Whether the curve is hidden initially")
    secret: Optional[bool] = Field(default=False, description="Hide expression text in calculator list")
    label: Optional[str] = Field(default=None, description="Label for point or expression")
    showLabel: Optional[bool] = Field(default=False, description="Show label on graph")
    sliderBounds: Optional[SliderBounds] = Field(default=None, description="Slider domain bounds if expression defines a parameter")

class LLMVisualizationResponse(BaseModel):
    title: str = Field(..., description="Title of the calculus concept visualization")
    concept_explanation: str = Field(..., description="Educational explanation of the visual representation")
    expressions: List[DesmosExpression] = Field(..., description="List of Desmos expressions to be injected")
    agent_steps: Optional[List[str]] = Field(default=None, description="Step-by-step agentic execution trace")

class VisualizeRequest(BaseModel):
    prompt: str = Field(..., min_length=2, max_length=500, description="Natural language calculus prompt e.g., 'Riemann sum of x^2'")
    
    @field_validator('prompt')
    @classmethod
    def prompt_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Prompt cannot be empty or whitespace only.')
        return v.strip()

class VisualizeAPIResponse(BaseModel):
    success: bool
    data: Optional[LLMVisualizationResponse] = None
    error: Optional[str] = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    processing_time_ms: Optional[float] = None
    agent_steps: Optional[List[str]] = None

class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    service: str = "Calculus Visualizer API"
    version: str = "1.0.0"
    timestamp: float = Field(default_factory=time.time)
    dependencies: Dict[str, str] = Field(default_factory=dict)

class JWTSessionToken(BaseModel):
    session_id: str
    created_at: float
    exp: float
