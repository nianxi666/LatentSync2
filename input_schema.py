# input_schema.py
from pydantic import BaseModel, Field
from typing import Optional

class Item(BaseModel):
    """
    Defines the expected input schema for the LatentSync model deployed on Inferless.
    Adjust fields based on the actual inputs LatentSync requires.
    """
    prompt: str = Field(
        ..., # '...' 表示这是必填字段
        example="A stunning photograph of a majestic wolf in a snowy forest, hyperrealistic, award-winning",
        description="Text prompt to guide the image generation."
    )
    num_inference_steps: Optional[int] = Field(
        default=50,
        example=50,
        description="Number of denoising steps. Higher values generally improve quality but take longer.",
        gt=0, # Greater than 0
        le=200 # Less than or equal to 200 (adjust based on model recommendations)
    )
    guidance_scale: Optional[float] = Field(
        default=7.5,
        example=7.5,
        description="Classifier-Free Guidance scale. Controls how much the image should conform to the prompt. Higher values are stricter.",
        gt=0, # Greater than 0
        le=25 # Less than or equal to 25 (adjust based on model recommendations)
    )
    # seed: Optional[int] = Field(
    #     default=None, # None通常表示随机种子
    #     example=12345,
    #     description="Seed for random number generation to ensure reproducibility. If None, a random seed will be used."
    # )

    # --- 根据 LatentSync 的具体需求添加更多参数 ---
    # 例如，如果 LatentSync 需要输入图像 (如用于个性化或 img2img):
    # input_image_base64: Optional[str] = Field(
    #     default=None,
    #     description="Base64 encoded input image (e.g., for img2img, personalization reference)."
    # )
    # strength: Optional[float] = Field(
    #     default=0.8,
    #     description="For img2img, controls the amount of noise added to the input image (0.0 means no noise, 1.0 means full noise).",
    #     ge=0.0, # Greater than or equal to 0.0
    #     le=1.0  # Less than or equal to 1.0
    # )
    # negative_prompt: Optional[str] = Field(
    #     default=None,
    #     example="ugly, blurry, low quality, deformed",
    #     description="Text prompt to guide what NOT to generate."
    # )

    # 示例配置，为 Inferless UI 提供一个默认的测试 JSON
    class Config:
        schema_extra = {
            "example": {
                "prompt": "A beautiful oil painting of a serene landscape at sunset",
                "num_inference_steps": 40,
                "guidance_scale": 7.0,
                # "seed": 42
            }
        }
