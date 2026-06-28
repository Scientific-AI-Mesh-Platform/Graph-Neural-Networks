FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
CMD ["python", "-c", "import mesh_hub; print('mesh_hub installed successfully.')"]
