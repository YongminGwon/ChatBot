FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY requirements.txt .

RUN uv pip install --no-cache-dir -r requirements.txt --system
RUN uv pip install --upgrade pip
RUN uv pip install --upgrade transformers datasets[audio] accelerate
RUN uv choco install ffmpeg-full -y

COPY ./app /app

CMD ["python", "main.py"]