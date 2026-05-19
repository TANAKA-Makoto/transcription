#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="transcription"
ASSETS_DIR="${ASSETS_DIR:-$(pwd)/assets}"
ENV_FILE="${ENV_FILE:-$(pwd)/.env}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"

DOCKER_ENV_ARGS=()
GPU_ARGS=()

if [[ -f "${ENV_FILE}" ]]; then
  DOCKER_ENV_ARGS+=(--env-file "${ENV_FILE}")
fi

if [[ -z "${HUGGING_FACE_TOKEN:-}" && ! -f "${ENV_FILE}" ]]; then
  echo "HUGGING_FACE_TOKEN is required. Example:" >&2
  echo "  HUGGING_FACE_TOKEN=xxxxxxxx ./run_docker.sh" >&2
  echo "  # or place it in ${ENV_FILE}" >&2
  exit 1
fi

if [[ ! -d "${ASSETS_DIR}" ]]; then
  echo "Assets directory not found: ${ASSETS_DIR}" >&2
  exit 1
fi

if [[ "${USE_GPU:-1}" == "1" ]]; then
  GPU_ARGS+=(--gpus all)
fi

docker build -t "${IMAGE_NAME}" .

docker run --rm -it \
  "${GPU_ARGS[@]}" \
  "${DOCKER_ENV_ARGS[@]}" \
  -e NETWORKX_BACKENDS= \
  -e NETWORKX_BACKEND= \
  -e NX_BACKENDS= \
  -e NX_BACKEND= \
  -e NETWORKX_AUTOMATIC_BACKENDS= \
  -e NETWORKX_BACKEND_PRIORITY= \
  -e NETWORKX_BACKEND_PRIORITY_ALGOS= \
  -e NETWORKX_BACKEND_PRIORITY_GENERATORS= \
  ${HUGGING_FACE_TOKEN:+-e HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN}"} \
  -v "${ASSETS_DIR}:/app/assets" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "${IMAGE_NAME}"
