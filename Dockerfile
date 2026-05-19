FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false

# Prevent networkx from parsing invalid backend names from env.
ENV NETWORKX_BACKENDS=

ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.10/site-packages/nvidia/cusparse/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel pkginfo

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        binutils \
        ffmpeg \
        git \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"
RUN pip install --no-cache-dir "nvidia-cudnn-cu12==9.1.0.70"

COPY pyproject.toml poetry.lock README.md ./
RUN poetry lock --no-update \
    && poetry install --no-interaction --no-ansi --only main

RUN python - <<'PY'
from pathlib import Path
import site

for sp in site.getsitepackages():
    for entry_path in Path(sp).glob("networkx-*.dist-info/entry_points.txt"):
        text = entry_path.read_text()
        lines = [line for line in text.splitlines() if not line.startswith("nx-loopback")]
        entry_path.write_text("\n".join(lines) + "\n")
PY

RUN find /usr/local/lib/python3.10/site-packages \
        -path '*/ctranslate2*' -type f -name '*.so*' -print0 \
    | xargs -0 -r patchelf --clear-execstack

COPY src ./src
COPY .flake8 ./

ENV PYTHONPATH=/app/src

CMD ["python", "./src/app.py"]
