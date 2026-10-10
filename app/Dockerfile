# EmoVision — image de production (CPU)
# Construction depuis la racine du dépôt :
#   docker build -f app/Dockerfile -t emovision .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY app/requirements.txt app/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r app/requirements.txt

# Code : définitions des modèles du mémoire + application
COPY models/ models/
COPY app/__init__.py app/
COPY app/server/ app/server/
COPY app/web/ app/web/
COPY app/tools/ app/tools/

# Poids : fournis dans app/weights/ (voir app/README.md) ou téléchargés à la construction
COPY app/weights/ app/weights/
ARG WEIGHTS_URL=https://github.com/bouramah/vit-fer-thesis/releases/download/v1.0/emovision_weights.pt
ARG YUNET_URL=https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
RUN test -f app/weights/emovision_weights.pt || curl -fL -o app/weights/emovision_weights.pt "$WEIGHTS_URL"; \
    test -f app/weights/face_detection_yunet_2023mar.onnx || curl -fL -o app/weights/face_detection_yunet_2023mar.onnx "$YUNET_URL"

ENV EMOVISION_DATA=/data PORT=8000
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://localhost:${PORT}/api/health || exit 1
CMD ["sh", "-c", "uvicorn app.server.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers"]
