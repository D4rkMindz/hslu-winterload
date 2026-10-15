import sys

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(find_dotenv())

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
    )

    entsoe_api_key: str = Field(alias="ENTSOE_API_KEY")


try:
    settings = Settings()
except ValidationError as e:
    print(f"Configuration error:\n{e}", file=sys.stderr)
    sys.exit(1)
