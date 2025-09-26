FROM python:3.11-slim

# Diretório de trabalho
WORKDIR /app

# Copia os arquivos
COPY . .

# Instala libs do sistema (necessárias para PyMuPDF, unstructured, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libmagic-dev \
    libgl1-mesa-glx \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

# Expõe a porta da FastAPI
EXPOSE 7860

# Comando para rodar o app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
