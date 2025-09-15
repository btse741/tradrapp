FROM python:3.11.9-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN mkdir -p /app/data /app/output

# Copy and set permissions for the run_all.sh script
COPY run_all.sh /app/
RUN chmod +x /app/run_all.sh

# Run the shell script to execute both Python scripts sequentially
CMD ["/app/run_all.sh"]
