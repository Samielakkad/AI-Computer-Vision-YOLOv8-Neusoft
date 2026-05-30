# -------------------------------------------------------------------------- #
#   Dockerfile — image reproductible pour entraîner / inférer le détecteur
# -------------------------------------------------------------------------- #
FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales pour OpenCV / Pillow.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU d'abord (image plus légère), puis le reste.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Par défaut on lance l'inférence ; voir predict.py pour les modes
# (image / vidéo / fps / dir / heatmap / export ONNX).
CMD ["python", "predict.py"]
