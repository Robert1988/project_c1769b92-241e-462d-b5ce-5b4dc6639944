FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    gcc \
    make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install wheel

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy function code
COPY experiment.py .
COPY main.py .

CMD ["python", "main.py"]
        