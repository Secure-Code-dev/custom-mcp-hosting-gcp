# GitHub MCP Server on Google Cloud Run

A **GitHub MCP (Model Context Protocol) Server** built with FastMCP and designed to run on **Google Cloud Run**. This server provides a comprehensive set of tools for interacting with GitHub repositories through the GitHub API, along with basic arithmetic operations.

## 🚀 Features

- **GitHub API Integration**: Full authentication and API interaction
- **Cloud Run Ready**: Configured for Google Cloud Run hosting
- **Comprehensive Logging**: Structured logging for debugging and monitoring
- **Async Operations**: Built with async/await for optimal performance
- **Error Handling**: Proper error handling for API failures

## 🛠️ Available Tools

### Basic Math Operations
- **`add(a, b)`**: Adds two integers
- **`subtract(a, b)`**: Subtracts two integers

### GitHub Repository Operations
- **`list_repositories(username)`**: Lists repositories for a user or authenticated user
- **`get_repository_contents(owner, repo, path)`**: Gets contents of a repository directory/file
- **`get_file_content(owner, repo, path)`**: Retrieves and decodes file content
- **`get_repository_languages(owner, repo)`**: Gets programming languages used in a repository

### GitHub Search & User Operations
- **`search_code(query, owner, repo)`**: Searches for code across repositories
- **`get_user_info(username)`**: Gets GitHub user information

## 📋 Prerequisites

- Python 3.8+
- GitHub Personal Access Token
- Google Cloud SDK (for deployment)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd custom-mcp-hosting-gcp
   ```

2. **Install dependencies**
   ```bash
   pip install fastmcp aiohttp python-dotenv
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GITHUB_TOKEN=your_github_personal_access_token
   PORT=8080
   ```

## 🔑 GitHub Token Setup

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with appropriate scopes:
   - `repo` (for repository access)
   - `user` (for user information)
   - `public_repo` (for public repository access)
3. Add the token to your `.env` file

## 🏃‍♂️ Running Locally

```bash
python server.py
```

The server will start on `http://0.0.0.0:8080` by default.

## ☁️ Deploying to Google Cloud Run

1. **Build and deploy**
   ```bash
   gcloud run deploy github-mcp-server \
     --source . \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars GITHUB_TOKEN=your_token_here
   ```

2. **Or use a Dockerfile** (create one if needed):
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "server.py"]
   ```

## 🏗️ Architecture

### Core Components

- **FastMCP Framework**: Creates the MCP server
- **GitHubClient**: Custom class for authenticated GitHub API requests
- **Async HTTP Client**: Uses `aiohttp` for HTTP requests
- **Environment Configuration**: Loads configuration from environment variables

### Code Structure

```
├── server.py           # Main server file
├── .env               # Environment variables (not in repo)
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 📖 API Usage Examples

### List Repositories
```python
# List authenticated user's repositories
await list_repositories()

# List specific user's repositories
await list_repositories("octocat")
```

### Get File Content
```python
# Get content of a specific file
content = await get_file_content("owner", "repo", "path/to/file.py")
print(content["content"])
```

### Search Code
```python
# Search across all accessible repositories
results = await search_code("function auth")

# Search in specific repository
results = await search_code("class GitHubClient", "owner", "repo")
```

## 🔒 Security

- **Token-based Authentication**: Uses GitHub Personal Access Tokens
- **Environment Variables**: Sensitive data stored in environment variables
- **Proper Headers**: Includes appropriate GitHub API headers and User-Agent

## 📊 Logging

The server includes comprehensive logging:
- Tool invocations with parameters
- API request details
- Error messages with status codes

## ⚠️ Error Handling

- **GitHub Token Validation**: Warns if `GITHUB_TOKEN` is not set
- **API Error Handling**: Catches and reports GitHub API errors with status codes
- **Type Safety**: Uses proper typing throughout the codebase

## 🔗 Dependencies

```txt
fastmcp>=0.1.0
aiohttp>=3.8.0
python-dotenv>=0.19.0
```

## 📝 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GITHUB_TOKEN` | GitHub Personal Access Token | Yes |
| `PORT` | Server port (default: 8080) | No |




---

**Note**: This server is designed to run on Google Cloud Run but can be deployed to any container platform that supports HTTP servers.

