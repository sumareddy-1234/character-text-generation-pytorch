FROM python:3.10-slim

WORKDIR /app

# install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir numpy matplotlib python-dotenv
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# copy project
COPY . .

# default command (can be overridden by docker-compose)
CMD ["python", "src/train.py", "--help"]