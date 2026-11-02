FROM python:3.12.9-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY src/python/main.py /app/main.py
ENTRYPOINT ["python", "/app/main.py"]
CMD ["--data-dir", "/app/data", "--output-dir", "/app/output"]
