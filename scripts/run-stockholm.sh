#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
if command -v caffeinate >/dev/null 2>&1; then
    set -- caffeinate -i .venv/bin/python
else
    set -- .venv/bin/python
fi

exec "$@" -u -m beygir run stockholm-extended-main \
    --dataset benchmark/extended_items.jsonl --split main --workers 4 --keep-waiting \
    --models aws-oss20 aws-oss120 aws-qwen32 aws-glmflash aws-nova-lite aws-devstral aws-glm47 aws-deepseek
