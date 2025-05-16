# input_schema.py
from pydantic import BaseModel, Field  # <--- 再次确认此导入！
from typing import Optional

class Item(BaseModel):
    """
    Defines the expected input schema for the LatentSync model's 'infer' method.
    """
    prompt: str = Field(
        ..., # '...' 表示这是必填字段
        example="A personalized portrait using LatentSync",
        description="Text prompt to guide the image generation."
    )
    num_inference_steps: Optional[int] = Field(
        default=50, # 与 app.py 中 get 的默认值一致
        example=50,
        description="Number of denoising steps.",
        gt=0,
        le=150
    )
    guidance_scale: Optional[float] = Field(
        default=7.5, # 与 app.py 中 get 的默认值一致
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
    # height: Optional[int] = Field(
    #     default=512,
    #     example=512,
    #     description="Height of the generated image."
    # )
    # width: Optional[int] = Field(
    #     default=512,
    #     example=512,
    #     description="Width of the generated image."
    # )
    # 根据您的 LatentSync 模型实际需要的输入参数添加更多字段
