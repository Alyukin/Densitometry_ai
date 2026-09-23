"""Подтверждение и исправление вердикта специалистом.

ТЗ (раздел «Дополнительно»): автоматическая коррекция разметки с подтверждением
специалиста. Здесь реализована вторая половина — врач видит предложенный сервисом
вердикт и либо соглашается с ним, либо исправляет.

Два правила, на которых держится вся конструкция:

1. **Автоматический вердикт не переписывается.** Решение врача пишется в отдельные
   поля. Иначе через неделю нельзя будет ответить на вопрос «а что предлагал сервис»,
   и метрики качества модели окажутся посчитаны по исправленным человеком данным.
2. **Исправление проверяется по тем же закрытым спискам, что и автоматический
   вердикт.** Врач не может ввести нарушение, которого нет в перечне заказчика, или
   приписать бедру нарушение позвоночника.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from app.models import REVIEW_CONFIRMED, REVIEW_CORRECTED, REVIEW_NONE, ImageResult
from app.models.study import utcnow
from app.processing.dxaqc.rules import CLOSED_VIOLATIONS

# Ввод врача проверяется по тем же закрытым спискам, что и вывод сервиса.
ALLOWED_VIOLATIONS = CLOSED_VIOLATIONS


def allowed_for(region: str | None) -> tuple[str, ...]:
    return ALLOWED_VIOLATIONS.get(region or "", ())


def validate_violations(region: str | None, violations: list[str]) -> list[str]:
    """Проверяет список нарушений по закрытому списку своей области."""
    allowed = allowed_for(region)
    if not allowed:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Область не определена — исправлять нарушения нечему. Сначала нужна успешная обработка.",
        )
    clean: list[str] = []
    for v in violations:
        name = (v or "").strip()
        if not name:
            continue
        if name not in allowed:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"«{name}» нет в списке нарушений для области «{region}». Допустимо: {', '.join(allowed)}",
            )
        if name not in clean:
            clean.append(name)
    return sorted(clean)


def apply_review(
    result: ImageResult,
    action: str,
    violations: list[str] | None = None,
    reviewer: str | None = None,
    comment: str | None = None,
) -> ImageResult:
    """`confirm` — согласиться с сервисом, `correct` — заменить вердикт, `reset` — снять проверку."""
    if action == "reset":
        result.review_status = REVIEW_NONE
        result.reviewed_quality_class = None
        result.reviewed_violation_type = None
        result.reviewed_by = None
        result.review_comment = None
        result.reviewed_at = None
        return result

    if result.processing_status != "success":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Изображение не обработано: подтверждать или исправлять нечего",
        )

    if action == "confirm":
        result.review_status = REVIEW_CONFIRMED
        # дублируем автоматический вердикт: дальше по нему считается согласие
        result.reviewed_quality_class = result.quality_class
        result.reviewed_violation_type = result.violation_type or ""
    elif action == "correct":
        clean = validate_violations(result.anatomical_region, violations or [])
        result.review_status = REVIEW_CORRECTED
        result.reviewed_quality_class = "1" if clean else "0"
        result.reviewed_violation_type = ";".join(clean)
    else:  # pragma: no cover — отсекается схемой
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Неизвестное действие: {action}")

    result.reviewed_by = (reviewer or "").strip() or None
    result.review_comment = (comment or "").strip() or None
    result.reviewed_at = utcnow()
    return result


def agreement(results: list[ImageResult]) -> dict:
    """Насколько сервис сходится с врачом — по тем строкам, которые врач посмотрел."""
    checked = [r for r in results if r.review_status in (REVIEW_CONFIRMED, REVIEW_CORRECTED)]
    same_class = sum(
        1 for r in checked if r.review_status == REVIEW_CONFIRMED or r.reviewed_quality_class == r.quality_class
    )
    return {
        "всего": len(results),
        "проверено": len(checked),
        "подтверждено": sum(1 for r in checked if r.review_status == REVIEW_CONFIRMED),
        "исправлено": sum(1 for r in checked if r.review_status == REVIEW_CORRECTED),
        "совпал_класс_качества": same_class,
        "согласие": round(same_class / len(checked), 4) if checked else None,
    }
