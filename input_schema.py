# input_schema.py
from pydantic import BaseModel, Field  # <--- 确保这一行是正确的！
from typing import Optional

class Item(BaseModel):
    """
    Defines the expected input schema for the LatentSync model.
    """
    prompt: str = Field(
        ..., 
        example="A futuristic city with flying cars, concept art",
        description="Text prompt to guide the image generation."
    )
    num_inference_steps: Optional[int] = Field(
        default=50,
        example=50,
        description="Number of denoising steps.",
        gt=0,
        le=150
    )
    guidance_scale: Optional[float] = Field(
        default=7.5,
        example=7.5,
        description="Classifier-free guidance scale.",
        gt=0,
        le=20
    )
    # seed: Optional[int] = Field(
    #     default=None,
    #     example=42,
    #     description="Seed for reproducibility. If None, a random seed will be used."
    # )
    # ... (根据您的模型实际需要的其他字段) ...

    # 示例配置，说明如何使用 Config 类（可选）
    # class Config:
    #     schema_extra = {
    #         "example": {
    #             "prompt": "A beautiful personalized image of a cat wearing a hat",
    #             "num_inference_steps": 50,
    #             "guidance_scale": 7.0
    #         }
    #     }
