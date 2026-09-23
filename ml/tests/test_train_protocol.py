"""Протокол v2: эпоха выбирается внутри train-части, внешний фолд в выборе не участвует.

Если внешний фолд просочится во внутреннюю validation или в pos_weight, программа не
упадёт — просто метрика тихо станет оптимистичной. Поэтому это проверяется отдельно.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from train.dataset import read_dataset  # noqa: E402
from train.train import EarlyStopping, inner_split  # noqa: E402

DATA = ROOT / "ml" / "data" / "processed" / "dataset.csv"


class _S:
    def __init__(self, i: int, fold: int, study: str) -> None:
        self.image_id, self.fold, self.study_dir = f"img{i}", fold, study


def _samples(n_folds: int = 5, per_fold: int = 4) -> list[_S]:
    return [_S(f * 100 + j, f, f"study{f}_{j // 2}") for f in range(n_folds) for j in range(per_fold)]


@pytest.mark.parametrize("fold", range(5))
def test_inner_split_never_touches_external_fold(fold: int) -> None:
    samples = _samples()
    train_part = [s for s in samples if s.fold != fold]
    fit, val, inner = inner_split(train_part, fold, 5)
    assert inner == (fold + 1) % 5
    assert all(s.fold != fold for s in fit + val)
    assert {s.fold for s in val} == {inner}
    assert {s.fold for s in fit} == set(range(5)) - {fold, inner}
    assert len(fit) + len(val) == len(train_part)


def test_inner_split_keeps_studies_apart() -> None:
    samples = _samples()
    for fold in range(5):
        fit, val, _ = inner_split([s for s in samples if s.fold != fold], fold, 5)
        assert not {s.study_dir for s in fit} & {s.study_dir for s in val}


@pytest.mark.skipif(not DATA.exists(), reason="нет data/processed (собирается dxa.build_dataset)")
def test_inner_split_on_real_folds_is_by_study() -> None:
    samples = read_dataset(DATA, DATA.parent)
    for fold in range(5):
        train_part = [s for s in samples if s.fold != fold]
        fit, val, _ = inner_split(train_part, fold, 5)
        external = {s.study_dir for s in samples if s.fold == fold}
        assert not {s.study_dir for s in fit} & {s.study_dir for s in val}
        assert not ({s.study_dir for s in fit} | {s.study_dir for s in val}) & external
        assert len(val) >= 30  # во внутренней validation есть на чём мерить loss


def test_early_stopping_keeps_best_epoch_and_stops_after_patience() -> None:
    es = EarlyStopping(patience=3)
    losses = [1.0, 0.8, 0.9, 0.7, 0.75, 0.71, 0.70]
    stopped_at = None
    for epoch, loss in enumerate(losses):
        es.step(loss, epoch)
        if es.stop:
            stopped_at = epoch
            break
    assert es.best_epoch == 3 and es.best == 0.7
    assert stopped_at == 6  # 0.70 — не строгое улучшение, третья эпоха без улучшения


def test_early_stopping_state_roundtrip() -> None:
    es = EarlyStopping(patience=2)
    es.step(1.0, 0)
    es.step(1.1, 1)
    again = EarlyStopping(patience=2)
    again.load(es.state())
    assert (again.best, again.best_epoch, again.bad) == (1.0, 0, 1)
    again.step(1.2, 2)
    assert again.stop
