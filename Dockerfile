FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Base distroless stage
FROM gcr.io/distroless/python3:latest as base
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Server stage
FROM base as server
COPY server.py ./
# Copy all Python files and directories to the server
COPY *.py ./
# Only copy directories that might exist
COPY . ./
EXPOSE 8080
CMD ["server.py"]

# Client stage
FROM base as client
COPY MCP_client.py ./
# Copy all Python files and directories to the client
COPY *.py ./
# Copy everything to ensure all dependencies are available
COPY . ./
EXPOSE 8000
CMD ["MCP_client.py"]