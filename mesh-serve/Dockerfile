FROM python:3.9-slim
WORKDIR /app
COPY mesh-core/ mesh-core/
COPY mesh-ml/ mesh-ml/
COPY mesh-sim/ mesh-sim/
COPY mesh-vis/ mesh-vis/
COPY mesh-hub/ mesh-hub/
COPY mesh-serve/ mesh-serve/
RUN pip install --no-cache-dir -e ./mesh-core -e ./mesh-ml -e ./mesh-sim -e ./mesh-vis -e ./mesh-hub -e ./mesh-serve
EXPOSE 8501
CMD ["streamlit", "run", "mesh-serve/mesh_serve/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
