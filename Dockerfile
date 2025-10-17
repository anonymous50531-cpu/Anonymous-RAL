#FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel
# Install system dependencies
RUN apt-get update && apt-get install -y git

COPY requirements.txt .

# Install Hugging Face & TRL ecosystem
RUN pip install --no-cache-dir -r requirements.txt

#avoid trition error
RUN mkdir -p /root/.triton/autotune && chmod -R 777 /root/.triton/autotune

WORKDIR /workspace