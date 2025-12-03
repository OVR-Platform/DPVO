#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_ARCHIVE="${ROOT_DIR}/models.zip"

if [ ! -f "${MODELS_ARCHIVE}" ]; then
    echo "Downloading DPVO models..."
    wget -O "${MODELS_ARCHIVE}" "https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip"
else
    echo "Reusing existing models archive at ${MODELS_ARCHIVE}."
fi

if [ -d "${ROOT_DIR}/models" ]; then
    echo "Overwriting existing models directory with archive contents."
fi

unzip -o "${MODELS_ARCHIVE}" -d "${ROOT_DIR}"

echo "Model download complete."
