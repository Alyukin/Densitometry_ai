"""Application settings loaded from environment variables / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore")

    # --- General ---
    app_name: str = "Densitometry AI"
    app_version: str = "0.1.0"
    environment: str = Field(default="development", description="development | production | test")
    log_level: str = "INFO"

    # --- Storage ---
    data_dir: Path = Path("./data")
    database_url: str | None = Field(
        default=None,
        description="SQLAlchemy URL. Defaults to sqlite:///<DATA_DIR>/densitometry.db",
    )

    # --- Upload limits ---
    max_upload_size_mb: int = 1024
    max_files_per_upload: int = 500
    max_images_per_study: int = Field(
        default=3, description="По ТЗ: до 3 изображений/серий на исследование. Превышение — предупреждение."
    )

    # --- Processing ---
    processor_backend: str = Field(default="mock", description="Имя зарегистрированного процессора: mock | ...")
    worker_concurrency: int = Field(default=2, ge=1, description="Количество параллельных задач обработки")
    processing_timeout_sec: int = Field(default=180, description="Лимит времени на исследование (ТЗ: ≤ 3 мин)")
    mock_delay_per_image_sec: float = Field(default=1.5, ge=0, description="Искусственная задержка mock-обработки")
    mock_seed: int = 42

    # --- CORS ---
    cors_origins: str = "http://localhost:5173,http://localhost:8080"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'densitometry.db').resolve()}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.results_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
