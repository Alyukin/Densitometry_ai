#!/usr/bin/env bash
# Baseline целиком: зависимости -> проверка разметки -> датасет -> обучение 5 фолдов -> метрики.
#
#   bash ml/run_baseline.sh
#
# Первый аргумент (или переменная DATA_ROOT) — папка Densitometry_data.
# Скрипт ничего не пишет в исходные данные, всё складывает в ml/data и ml/runs.
set -euo pipefail

cd "$(dirname "$0")"

DATA_ROOT="${1:-${DATA_ROOT:-../../Densitometry_data}}"
OUT_DATA="${OUT_DATA:-data/processed}"
OUT_RUN="${OUT_RUN:-runs/baseline}"
EPOCHS="${EPOCHS:-40}"
BACKBONE="${BACKBONE:-resnet18}"
BATCH="${BATCH:-16}"

if [ ! -d "$DATA_ROOT/НД_для_обучения/Исследования" ]; then
  echo "Не нашёл выгрузку в: $DATA_ROOT" >&2
  echo "Запустите так:  bash ml/run_baseline.sh /путь/к/Densitometry_data" >&2
  exit 1
fi

echo "== 1/3 зависимости =="
python3 -m pip install -q -r requirements.txt
python3 -c 'import torch' 2>/dev/null || {
  echo "ставлю torch под CUDA (H200 — cu126)"
  python3 -m pip install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
}
python3 - <<'PY'
import torch
print(f"torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'НЕТ — обучение пойдёт на CPU'}")
PY

echo
echo "== 2/3 проверка разметки и сборка датасета =="
python3 -m dxa.build_dataset --data-root "$DATA_ROOT" --out "$OUT_DATA"
python3 -m dxa.contact_sheet --data "$OUT_DATA" --out qa || true

echo
echo "== 3/3 обучение: 5 фолдов по исследованиям =="
# веса ImageNet тянутся один раз; если сети нет — сразу понятно, в чём дело
python3 - "$BACKBONE" <<'PY' || { echo "Не скачались веса ImageNet. Либо дайте машине доступ к download.pytorch.org, либо добавьте флаг --no-pretrained (качество будет заметно ниже)." >&2; exit 1; }
import sys
from train.model import build_backbone
build_backbone(sys.argv[1], pretrained=True)
print("веса ImageNet на месте")
PY

python3 -m train.train \
  --data "$OUT_DATA" \
  --out "$OUT_RUN" \
  --backbone "$BACKBONE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH"

echo
echo "Готово."
echo "  метрики (+95% ДИ):   $OUT_RUN/metrics.json"
echo "  предсказания OOF:    $OUT_RUN/oof_predictions.csv"
echo "  веса фолдов:         $OUT_RUN/fold*.pt"
echo "  отчёт по данным:     $OUT_DATA/report.md"
echo "  разметка: правки — $OUT_DATA/label_fixes.csv, спорное — $OUT_DATA/manual_review.csv"
echo "  контактные листы:    qa/*.png"
