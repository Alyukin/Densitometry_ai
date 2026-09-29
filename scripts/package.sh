#!/usr/bin/env bash
# Пакет для сдачи: образы Docker, compose без сборки из исходников, пороги, отчёты.
#
#   bash scripts/package.sh            # или: make package
#
# На целевой машине нужен только Docker с Compose v2 — исходники и интернет не нужны:
# образы уже содержат всё, включая модель второго мнения. Видеокарта не нужна.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}"
OUT="dist/densitometry-ai-$VERSION"
rm -rf "$OUT"
mkdir -p "$OUT/docs"

echo "== сборка образов"
docker compose build
IMAGES=$(docker compose config --images | sort -u | tr '\n' ' ')
echo "== сохранение образов: $IMAGES"
# shellcheck disable=SC2086
docker save $IMAGES | gzip -1 > "$OUT/images.tar.gz"

# compose без секций build: на целевой машине собирать нечего, образы уже загружены
python3 - docker-compose.yml "$OUT/docker-compose.yml" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
out, skip = [], None
for line in open(src, encoding="utf-8").read().splitlines():
    indent = len(line) - len(line.lstrip())
    if skip is not None and (not line.strip() or indent > skip):
        continue
    skip = None
    if re.match(r"^\s+build:\s*$", line):
        skip = indent
        continue
    out.append(line)
open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")
PY

cp .env.example "$OUT/"
cp backend/app/processing/dxaqc/thresholds.json "$OUT/"
cp README.md OVERVIEW.md "$OUT/docs/"
cp ml/baseline/metrics.json ml/baseline/localization_metrics.json "$OUT/docs/"

cat > "$OUT/INSTALL.txt" <<EOF
Densitometry AI $VERSION — контроль качества DXA-исследований

1. Загрузить образы:          docker load -i images.tar.gz
2. Настройки (необязательно): cp .env.example .env
3. Запустить:                 docker compose up -d
4. Сайт:                      http://<адрес машины>:8080 (API и Swagger — /docs)

Остановить — docker compose down, удалить все данные — docker compose down -v.
Вердикт выносят правила ТЗ (пороги — thresholds.json, он же внутри образа).
Оценка ИИ-модели для бедра — справочно, выключается SECOND_OPINION=false в .env.
Результаты не являются медицинским заключением. Подробности — docs/.
EOF

(cd "$OUT" && sha256sum images.tar.gz docker-compose.yml thresholds.json .env.example > SHA256SUMS)
echo "== готово: $OUT ($(du -sh "$OUT" | cut -f1))"
ls -la "$OUT"
