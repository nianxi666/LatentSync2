# input_schema.py
from pydantic import BaseModel, Field
from typing import Optional

class Item(BaseModel):
    """
    Defines the expected input schema for the LatentSync model.
    """
    prompt: str = Field(
        ..., # '...' 表示这是必填字段
        example="A futuristic city with flying cars, concept art",
        description="Text prompt to guide the image generation."
    )
    num_inference_steps: Optional[int] = Field(
        default=50,
        example=50,
        description="Number of denoising steps.",
        gt=0, # Greater than 0
        le=150 # Less than or equal to 150 (adjust as per model's capability)
    )
    guidance_scale: Optional[float] = Field(
        default=7.5,
        example=7.5,
        description="Classifier-free guidance scale. Higher values lead to images closer to the prompt, but may reduce diversity.",
        gt=0, # Greater than 0
        le=20 # Less than or equal to 20 (adjust as per model's capability)
    )
    # seed: Optional[int] = Field(
    #     default=None, # None通常表示随机种子
    #     example=42,
    #     description="Seed for reproducibility. If None, a random seed will be used."
    # )
    # 根据 LatentSync 项目的实际输入参数，您可以在这里添加更多字段。
    # 例如，如果它接受参考图像或控制图像：
    # reference_image_base64: Optional[str] = Field(
    #     default=None,
    #     description="Base64 encoded reference image for style transfer or personalization."
    # )
    # control_image_base64: Optional[str] = Field(
    #     default=None,
    #     description="Base64 encoded control image for ControlNet-like features."
    # )
    # strength: Optional[float] = Field(
    #     default=0.8,
    #     example=0.8,
    #     description="Strength of the control image or reference image influence.",
    #     ge=0, # Greater than or equal to 0
    #     le=1  # Less than or equal to 1
    # )

    # 示例配置，说明如何使用 Config 类（可选）
    # class Config:
    #     schema_extra = {
    #         "example": {
    #             "prompt": "A beautiful personalized image of a cat wearing a hat",
    #             "num_inference_steps": 50,
    #             "guidance_scale": 7.0
    #         }
    #     }
