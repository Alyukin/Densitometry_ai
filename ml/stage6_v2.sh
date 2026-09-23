#!/usr/bin/env bash
# Этап 6, протокол v2 (questions.md, К2): три заранее объявленные конфигурации на тех же
# 5 фолдах по исследованиям, что у baseline, и сводка с решением о следующем шаге.
#
#   bash ml/stage6_v2.sh /путь/к/Densitometry_data
#
# Что делает:
#   1. зависимости, GPU, предобученные веса (ImageNet и TorchXRayVision);
#   2. датасет из выгрузки + проверка, что фолды и метки те же, что у baseline;
#   3. измерения правил на тех же снимках (для сравнения);
#   4. Arak: подготовка кадров и предобучение xrv-densenet121 (15 эпох);
#   5. обучение: resnet18, xrv-densenet121, xrv-densenet121 + Arak —
#      число эпох выбирается по внутреннему фолду внутри train-части, затем модель учится
#      заново на всей train-части столько эпох; внешний фолд — только для предсказаний;
#   6. сводка runs/v2/SUMMARY.md и архив результатов без весов.
#
# Прерванный запуск продолжается той же командой: готовые фолды не пересчитываются.
# В исходные данные ничего не пишется. Настройки — переменными окружения (см. ниже);
# менять их для настоящего прогона не нужно, они для проверки скрипта на CPU.
#
# Окружение: если не активирован venv, скрипт сам создаёт ml/.venv и ставит туда всё
# нужное (Python >= 3.11). Свой venv/conda — активируйте его перед запуском.
#
# Память GPU: DenseNet121 на 384×320 при batch 16 требует ~5 ГБ только под активации.
# На картах меньше 12 ГБ (например, GTX 1660 Ti, 6 ГБ) все три конфигурации и
# предобучение идут в смешанной точности (AMP). Решение зависит только от объёма памяти,
# принимается до обучения и одно на все конфигурации — сравнение остаётся на равных.
set -euo pipefail

cd "$(dirname "$0")"

DATA_ROOT="${1:-${DATA_ROOT:-../../Densitometry_data}}"
OUT_DATA="${OUT_DATA:-data/processed}"
RUNS="${RUNS:-runs/v2}"
ARAK_DATA="${ARAK_DATA:-data/arak_full}"
DEVICE="${DEVICE:-cuda}"
NPROC="$(nproc 2>/dev/null || echo 4)"
WORKERS="${WORKERS:-$(( NPROC > 10 ? 8 : (NPROC > 3 ? NPROC - 2 : 1) ))}"
AMP="${AMP:-auto}"   # auto | 0 | 1
VENV="${VENV:-auto}" # auto — ml/.venv, если не активирован свой; 0 — не трогать окружение
# Протокол v2 — не менять для настоящего прогона
EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-8}"
BATCH="${BATCH:-16}"
HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-320}"
ARAK_EPOCHS="${ARAK_EPOCHS:-15}"
# Только для проверки скрипта
ONLY_FOLD="${ONLY_FOLD:--1}"
ARAK_LIMIT="${ARAK_LIMIT:-}"
SKIP_DEPS="${SKIP_DEPS:-0}"
ALLOW_OTHER_FOLDS="${ALLOW_OTHER_FOLDS:-0}"

say() { printf '\n== %s ==\n' "$*"; }

if [ ! -d "$DATA_ROOT/НД_для_обучения/Исследования" ]; then
  echo "Не нашёл выгрузку в: $DATA_ROOT" >&2
  echo "Запустите так:  bash ml/stage6_v2.sh /путь/к/Densitometry_data" >&2
  exit 1
fi
if [ ! -f "$DATA_ROOT/ArakDATA/TableFinal-OSTEO.xlsx" ]; then
  echo "Нет открытого набора Arak: $DATA_ROOT/ArakDATA (нужен для третьей конфигурации)" >&2
  exit 1
fi
mkdir -p "$RUNS"
exec > >(tee -a "$RUNS/stage6_v2.log") 2>&1
echo "старт: $(date -Iseconds)"

say "1/6 окружение, зависимости и GPU"
python3 - <<'PY' || exit 1
import sys
if sys.version_info < (3, 11):
    sys.exit(
        f"Нужен Python >= 3.11, сейчас {sys.version.split()[0]}. Например:\n"
        "  conda create -n dxa python=3.12 -y && conda activate dxa && VENV=0 bash ml/stage6_v2.sh ..."
    )
PY
if [ "$VENV" = "auto" ] && ! python3 -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)'; then
  if [ ! -x .venv/bin/python3 ]; then
    echo "создаю ml/.venv"
    python3 -m venv .venv || {
      echo "Не удалось создать venv. Ubuntu: sudo apt install python3-venv" >&2
      echo "или своё окружение: conda activate <env> && VENV=0 bash ml/stage6_v2.sh ..." >&2
      exit 1
    }
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  echo "окружение: $(python3 -c 'import sys; print(sys.prefix)')"
fi
if [ "$SKIP_DEPS" != "1" ]; then
  python3 -m pip install -q --upgrade pip
  python3 -m pip install -q -r requirements.txt
  python3 -c 'import torch' 2>/dev/null || {
    echo "ставлю torch под CUDA (H200 — cu126)"
    python3 -m pip install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
  }
  python3 -c 'import torchxrayvision' 2>/dev/null || python3 -m pip install -q torchxrayvision==1.5.4
fi
python3 - "$DEVICE" <<'PY'
import sys, torch
dev = sys.argv[1]
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if gpu else 0
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), GPU: {gpu or 'нет'}" + (f", {mem:.1f} ГБ" if gpu else ""))
if dev.startswith("cuda") and gpu is None:
    sys.exit(
        "GPU не виден, а DEVICE=cuda. Проверьте nvidia-smi: для torch с CUDA 12.6 нужен драйвер NVIDIA >= 525. "
        "На CPU полный прогон займёт больше суток."
    )
PY
if [ "$AMP" = "auto" ]; then
  AMP="$(python3 -c 'import torch; print(int(torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 12 * 2**30))')"
fi
AMP_ARGS=()
if [ "$AMP" = "1" ]; then
  AMP_ARGS=(--amp)
  echo "смешанная точность (AMP): да — для всех конфигураций и предобучения"
else
  echo "смешанная точность (AMP): нет, fp32"
fi
echo "потоков загрузки данных: $WORKERS"
# предобученные веса тянутся один раз (ImageNet — download.pytorch.org, TorchXRayVision — github.com)
python3 - <<'PY' || { echo "Не скачались предобученные веса: нужен доступ к download.pytorch.org и github.com (или положите файлы в ~/.cache/torch/hub/checkpoints и ~/.torchxrayvision/models_data)." >&2; exit 1; }
from train.model import build_backbone
for name in ("resnet18", "xrv-densenet121"):
    build_backbone(name, pretrained=True)
    print(f"веса {name} на месте")
PY

say "2/6 датасет и проверка фолдов"
python3 -m dxa.build_dataset --data-root "$DATA_ROOT" --out "$OUT_DATA"
if ! python3 -m train.summary --check-folds "$OUT_DATA/dataset.csv"; then
  if [ "$ALLOW_OTHER_FOLDS" != "1" ]; then
    echo "Фолды или метки отличаются от тех, на которых посчитаны baseline и первый прогон." >&2
    echo "Сравнение с ними потеряет смысл. Если данные изменились намеренно — ALLOW_OTHER_FOLDS=1." >&2
    exit 1
  fi
fi

say "3/6 измерения правил на тех же снимках"
python3 baseline/featdump.py --data "$OUT_DATA" --out "$RUNS/features.csv"

COMMON=(--data "$OUT_DATA" --epochs "$EPOCHS" --batch-size "$BATCH" --height "$HEIGHT" --width "$WIDTH"
  --early-stopping inner --patience "$PATIENCE" --refit --workers "$WORKERS" --device "$DEVICE"
  --only-fold "$ONLY_FOLD" --resume ${AMP_ARGS[@]+"${AMP_ARGS[@]}"})

say "4/6 конфигурация 1: resnet18 (ImageNet)"
python3 -m train.train "${COMMON[@]}" --backbone resnet18 --out "$RUNS/resnet18"

say "4/6 конфигурация 2: xrv-densenet121 (TorchXRayVision)"
python3 -m train.train "${COMMON[@]}" --backbone xrv-densenet121 --out "$RUNS/xrv-densenet121"

say "5/6 Arak: подготовка кадров и предобучение бэкбона"
if [ ! -f "$ARAK_DATA/arak.csv" ]; then
  python3 -m dxa.arak --data-root "$DATA_ROOT" --out "$ARAK_DATA" ${ARAK_LIMIT:+--limit "$ARAK_LIMIT"}
else
  echo "уже подготовлено: $ARAK_DATA"
fi
if [ ! -f "$RUNS/arak/pretrain_metrics.json" ]; then
  python3 -m train.pretrain --data "$ARAK_DATA" --out "$RUNS/arak" --backbone xrv-densenet121 \
    --epochs "$ARAK_EPOCHS" --batch-size "$BATCH" --height "$HEIGHT" --width "$WIDTH" \
    --workers "$WORKERS" --device "$DEVICE" ${AMP_ARGS[@]+"${AMP_ARGS[@]}"}
else
  echo "предобучение уже есть: $RUNS/arak/backbone.pt"
fi

say "5/6 конфигурация 3: xrv-densenet121 + предобучение на Arak"
python3 -m train.train "${COMMON[@]}" --backbone xrv-densenet121 --init-backbone "$RUNS/arak/backbone.pt" \
  --out "$RUNS/xrv-densenet121_arak"

say "6/6 сводка и решение по правилу К2"
REF=()
[ -f runs/cnn/oof_predictions.csv ] && REF=(--reference runs/cnn)
python3 -m train.summary --data "$OUT_DATA" --features "$RUNS/features.csv" --out "$RUNS" \
  --runs "$RUNS/resnet18" "$RUNS/xrv-densenet121" "$RUNS/xrv-densenet121_arak" ${REF[@]+"${REF[@]}"}

python3 - "$RUNS" "$AMP" <<'PY'
import json, platform, subprocess, sys, torch
gpu = torch.cuda.is_available()
env = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if gpu else None,
    "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if gpu else None,
    "amp": sys.argv[2] == "1",
    "python": platform.python_version(),
}
try:
    env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
except OSError:
    pass
open(f"{sys.argv[1]}/env.json", "w").write(json.dumps(env, ensure_ascii=False, indent=2))
PY

RUNS_ABS="$(cd "$RUNS" && pwd)"
ARCHIVE="${ARCHIVE:-$(pwd)/stage6_v2_results.tgz}"
tar czf "$ARCHIVE" --exclude='*.pt' --exclude='*.tmp' -C "$(dirname "$RUNS_ABS")" "$(basename "$RUNS_ABS")"
echo
echo "Готово: $(date -Iseconds)"
echo "  сводка и решение:  $RUNS_ABS/SUMMARY.md"
echo "  все цифры:         $RUNS_ABS/summary.json"
echo "  веса фолдов:       $RUNS_ABS/*/fold*.pt (в архив не входят)"
echo "  архив без весов:   $ARCHIVE — его и пришлите"
