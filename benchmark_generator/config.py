from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "BENCH_GEN_"}

    llm_model: str = "gpt-4o"
    llm_api_base: str = "https://api.openai.com/v1"
    llm_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    benchmark_root: Path = Path(".")
    output_dir: Path = Path("benchmark_generator/output")
    code_timeout: int = 300


@lru_cache
def get_settings() -> Settings:
    return Settings()
