"""Database engine / session factory (SQLAlchemy 2.0, SQLite by default)."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _configure_sqlite(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        url = get_settings().sqlalchemy_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        if url.startswith("sqlite"):
            _configure_sqlite(_engine)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def reset_engine() -> None:
    """Used by tests to re-create the engine with new settings."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def _add_missing_columns(engine: Engine) -> None:
    """Дописывает в существующие таблицы колонки, появившиеся в моделях позже.

    Полноценные миграции здесь избыточны: схема плоская, а данные — результаты
    обработки, которые можно пересчитать. Но молча терять уже загруженные
    исследования при обновлении сервиса нельзя, поэтому недостающие колонки
    добавляются на месте. Существующие колонки не трогаются никогда.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue  # таблицу целиком создаст create_all
            have = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in have or column.primary_key:
                    continue
                ddl = column.type.compile(engine.dialect)
                default = ""
                if column.default is not None and getattr(column.default, "is_scalar", False):
                    value = column.default.arg
                    default = f" DEFAULT {value!r}" if isinstance(value, str) else f" DEFAULT {value}"
                logger.info("Схема: добавляю колонку %s.%s", table.name, column.name)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl}{default}'))


def init_db() -> None:
    from app import models  # noqa: F401  (register models)

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _add_missing_columns(engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    get_engine()
    assert _SessionLocal is not None
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Session for background workers."""
    get_engine()
    assert _SessionLocal is not None
    db = _SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
