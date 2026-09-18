from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.services.task_runner import TaskRunner


def get_runner(request: Request) -> TaskRunner:
    return request.app.state.runner


DbDep = Annotated[Session, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
RunnerDep = Annotated[TaskRunner, Depends(get_runner)]
