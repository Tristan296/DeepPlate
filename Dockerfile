FROM python:3.10

# Install dependencies
COPY requirements.txt /tmp/
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && pip install -r /tmp/requirements.txt

# For display
ENV DISPLAY=host.docker.internal:0
WORKDIR /app
COPY . /app

CMD ["python3", "main.py"]
