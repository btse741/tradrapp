FROM python:3.11.9-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN mkdir -p /app/data /app/output

CMD ["python", "models/etf_momentum/data_update.py"]
