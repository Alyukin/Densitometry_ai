"""Закрытые списки заказчика заданы в сервисе один раз (`dxaqc.rules.CLOSED_VIOLATIONS`).

Две копии импортировать его не могут: подготовка данных в `ml/dxa` не должна зависеть
от backend, а интерфейс написан на TypeScript. Расхождение с ними не роняет программу —
врач просто не сможет выбрать нарушение или датасет соберётся с другим названием.
Поэтому копии сверяются здесь.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend" / "app" / "processing"))  # dxaqc без FastAPI

from dxaqc.rules import CLOSED_VIOLATIONS, REGION_FEMUR, REGION_SPINE  # noqa: E402

from dxa import labels  # noqa: E402

FRONTEND = ROOT / "frontend" / "src" / "components" / "StudyDrawer.tsx"


def test_customer_answer_to_question_6() -> None:
    """Ответ заказчика на вопрос 6, дословно."""
    assert CLOSED_VIOLATIONS == {
        "Поясничный отдел позвоночника": (
            "Некорректная укладка",
            "Не выравнена ось позвоночника",
            "Присутствуют посторонние предметы",
        ),
        "Проксимальный отдел бедра": ("Некорректная укладка", "Некорректная область интереса"),
    }


def test_dataset_preparation_uses_the_same_lists() -> None:
    assert (labels.REGION_SPINE, labels.REGION_FEMUR) == (REGION_SPINE, REGION_FEMUR)
    assert tuple(labels.SPINE_VIOLATIONS) == CLOSED_VIOLATIONS[REGION_SPINE]
    assert tuple(labels.FEMUR_VIOLATIONS) == CLOSED_VIOLATIONS[REGION_FEMUR]


def test_interface_offers_the_same_lists() -> None:
    if not FRONTEND.exists():
        pytest.skip("исходников интерфейса рядом нет")
    src = FRONTEND.read_text(encoding="utf-8")
    block = re.search(r"const ALLOWED_VIOLATIONS[^=]*=\s*\{(.*?)\n\};", src, re.S)
    assert block, "в StudyDrawer.tsx не найден ALLOWED_VIOLATIONS"
    found = {
        region: tuple(re.findall(r'"([^"]+)"', items))
        for region, items in re.findall(r'"([^"]+)":\s*\[(.*?)\]', block.group(1), re.S)
    }
    assert found == CLOSED_VIOLATIONS
