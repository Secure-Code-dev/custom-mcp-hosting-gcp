# Builder stage with UV
FROM python:3.11-slim as builder

# Install UV
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies using UV to a target directory
RUN uv pip install --target /opt/deps -r requirements.txt
# COPY ./requirements.txt ./uv.lock ./

# COPY uv.lock . 
# RUN uv pip install --target /opt/deps --from-lock uv.lock

# Base stage using python slim instead of distroless
FROM python:3.11-slim as base

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy the installed packages from builder
COPY --from=builder /opt/deps /opt/deps
ENV PYTHONPATH="/opt/deps:$PYTHONPATH"

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Server stage
FROM base as server
COPY --chown=appuser:appuser server.py ./
# Copy all Python files and directories to the server
COPY --chown=appuser:appuser *.py ./
# Copy everything to ensure all dependencies are available
COPY --chown=appuser:appuser . ./
EXPOSE 8080
CMD ["python", "server.py"]

# Client stage
FROM base as client
COPY --chown=appuser:appuser MCP_client.py ./
# Copy all Python files and directories to the client
COPY --chown=appuser:appuser *.py ./
# Copy everything to ensure all dependencies are available
COPY --chown=appuser:appuser . ./
EXPOSE 8000
CMD ["python", "MCP_client.py"]