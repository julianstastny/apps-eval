"""
Configuration management for the APPS evaluation service.
"""
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Execution settings
    max_memory_mb: int = 512
    default_timeout: float = 4.0
    
    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: str = "apps_eval.log"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    @property
    def max_memory_bytes(self) -> int:
        """Get maximum memory in bytes."""
        return self.max_memory_mb * 1024 * 1024


# Global settings instance
settings = Settings() 