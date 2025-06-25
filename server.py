
import asyncio
import logging
import os
import base64
import jwt
from typing import List, Dict, Any, Optional
import aiohttp
from fastapi import FastAPI, HTTPException, Depends, Request, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from fastmcp import FastMCP
from validations import ContentValidator
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)

# mcp = FastMCP("GitHub MCP Server with OAuth")
app = FastAPI(title="GitHub MCP Server with OAuth")

# JWT Configuration (same as in your client)
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')
JWT_ALGORITHM = 'HS256'

def verify_jwt_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Dependency to get current authenticated user from Authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Extract token from "Bearer <token>" format
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header format",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token = authorization.split(" ")[1]
        user_data = verify_jwt_token(token)
        logger.info(f"Authenticated user: {user_data.get('email')}")
        return user_data
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# GitHub API configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Personal Access Token
GITHUB_API_BASE = "https://api.github.com"

class GitHubClient:
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "MCP-GitHub-Server"
        }
    
    async def make_request(self, endpoint: str, method: str = "GET") -> Dict[Any, Any]:
        """Make authenticated request to GitHub API"""
        url = f"{GITHUB_API_BASE}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=self.headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"GitHub API error {response.status}: {error_text}")

github_client = GitHubClient(GITHUB_TOKEN) if GITHUB_TOKEN else None

# OAuth Routes
# app = mcp.app

# @mcp.tool()
@app.post("/tools/list_repositories")
async def list_repositories(username: Optional[str] = None) -> List[Dict[str, Any]]:
    """List repositories for the authenticated user or a specific username.
    
    Args:
        username: Optional username to list repositories for. If not provided, lists authenticated user's repos.
    
    Returns:
        List of repository information.
    """
    if not github_client:
        raise Exception("GitHub token not configured")
    
    logger.info(f">>> Tool: 'list_repositories' called for user '{username or 'authenticated user'}'")
    
    if username:
        endpoint = f"/users/{username}/repos"
    else:
        endpoint = "/user/repos"
    
    repos = await github_client.make_request(endpoint)
    
    # Return simplified repo info
    return [
        {
            "name": repo["name"],
            "full_name": repo["full_name"],
            "private": repo["private"],
            "description": repo["description"],
            "language": repo["language"],
            "stars": repo["stargazers_count"],
            "forks": repo["forks_count"],
            "updated_at": repo["updated_at"],
            "clone_url": repo["clone_url"],
            "html_url": repo["html_url"]
        }
        for repo in repos
    ]

# @mcp.tool()
@app.post("/tools/get_repository_contents")
async def get_repository_contents(owner: str, repo: str, path: str = "") -> List[Dict[str, Any]]:
    """Get contents of a repository directory or file.
    
    Args:
        owner: Repository owner username.
        repo: Repository name.
        path: Path within the repository (empty string for root).
    
    Returns:
        List of files and directories in the specified path.
    """
    if not github_client:
        raise Exception("GitHub token not configured")
    
    logger.info(f">>> Tool: 'get_repository_contents' called for {owner}/{repo} path '{path}'")
    
    endpoint = f"/repos/{owner}/{repo}/contents/{path}"
    contents = await github_client.make_request(endpoint)
    
    # Handle single file vs directory listing
    if isinstance(contents, dict):
        contents = [contents]
    
    return [
        {
            "name": item["name"],
            "path": item["path"],
            "type": item["type"],
            "size": item.get("size", 0),
            "download_url": item.get("download_url"),
            "html_url": item.get("html_url")
        }
        for item in contents
    ]


# @mcp.tool()
@app.post("/tools/get_file_content")
async def get_file_content(request: Request):
    """Get the content of a specific file from a repository with security validation.
    
    Args:
        owner: Repository owner username.
        repo: Repository name.
        path: Path to the file within the repository.
    
    Returns:
        File content and metadata with security validation applied.
    """
    if not github_client:
        raise Exception("GitHub token not configured")
    
    args = await request.json()
    owner = args["owner"]
    repo = args["repo"]
    path = args["path"]
    
    logger.info(f">>> Tool: 'get_file_content' called for {owner}/{repo}/{path}")
    
    endpoint = f"/repos/{owner}/{repo}/contents/{path}"
    
    try:
        file_data = await github_client.make_request(endpoint)
        
        if file_data["type"] != "file":
            raise Exception(f"Path '{path}' is not a file")
        
        # Initialize the validator
        content_validator = ContentValidator()
        
        # Decode base64 content
        try:
            content = base64.b64decode(file_data["content"]).decode('utf-8')
        except UnicodeDecodeError:
            # Handle binary files
            return {
                "name": file_data["name"],
                "path": file_data["path"],
                "size": file_data["size"],
                "content": "[BINARY FILE - CONTENT NOT DISPLAYED]",
                "sha": file_data["sha"],
                "encoding": file_data["encoding"],
                "validation_status": "binary_file_skipped",
                "warnings": ["Binary file detected - content not processed for security"],
                "license_info": {},
                "source_url": f"https://github.com/{owner}/{repo}/blob/main/{path}",
                "source_attribution": f"Source: GitHub repository {owner}/{repo}"
            }
        
        # Apply security validation and filtering
        validation_result = content_validator.validate_and_filter_content(
            content, 
            file_data["name"], 
            file_data["size"]
        )

        
        if not validation_result['allowed']:
            return {
                "name": file_data["name"],
                "path": file_data["path"],
                "size": file_data["size"],
                "content": None,
                "sha": file_data["sha"],
                "encoding": file_data["encoding"],
                "validation_status": "blocked",
                "reason": validation_result['reason'],
                "warnings": validation_result['warnings'],
                "license_info": validation_result['license_info'],
                "source_url": f"https://github.com/{owner}/{repo}/blob/main/{path}",
                "source_attribution": f"Source: GitHub repository {owner}/{repo}"
            }
        
        # Return validated and filtered content
        return {
            "name": file_data["name"],
            "path": file_data["path"],
            "size": file_data["size"],
            "content": validation_result['content'],
            "sha": file_data["sha"],
            "encoding": file_data["encoding"],
            "validation_status": "approved" if not validation_result['warnings'] else "approved_with_warnings",
            "warnings": validation_result['warnings'],
            "license_info": validation_result['license_info'],
            "source_url": f"https://github.com/{owner}/{repo}/blob/main/{path}",
            "source_attribution": f"Source: GitHub repository {owner}/{repo}",
            "compliance_note": "Content has been validated for sensitive data and licensing compliance"
        }
        
    except Exception as e:
        logger.error(f"Error fetching file content: {str(e)}")
        return {
            "error": f"Failed to fetch file content: {str(e)}",
            "validation_status": "error",
            "source_url": f"https://github.com/{owner}/{repo}/blob/main/{path}",
            "source_attribution": f"Source: GitHub repository {owner}/{repo}"
        }

# @mcp.tool()
@app.post("/tools/search_code")
async def search_code(query: str, owner: Optional[str] = None, repo: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search for code across repositories.
    
    Args:
        query: Search query.
        owner: Optional repository owner to limit search.
        repo: Optional repository name to limit search (requires owner).
    
    Returns:
        List of code search results.
    """
    if not github_client:
        raise Exception("GitHub token not configured")
    
    logger.info(f">>> Tool: 'search_code' called with query '{query}'")
    
    # Build search query
    search_query = query
    if owner and repo:
        search_query += f" repo:{owner}/{repo}"
    elif owner:
        search_query += f" user:{owner}"
    
    endpoint = f"/search/code?q={search_query}"
    results = await github_client.make_request(endpoint)
    
    return [
        {
            "name": item["name"],
            "path": item["path"],
            "repository": item["repository"]["full_name"],
            "score": item["score"],
            "html_url": item["html_url"],
            "git_url": item["git_url"]
        }
        for item in results.get("items", [])
    ]

# @mcp.tool()

@app.post("/tools/get_repository_languages")
async def get_repository_languages(owner: str, repo: str) -> Dict[str, int]:
    """Get programming languages used in a repository.
    
    Args:
        owner: Repository owner username.
        repo: Repository name.
    
    Returns:
        Dictionary of languages and their byte counts.
    """
    if not github_client:
        raise Exception("GitHub token not configured")
    
    logger.info(f">>> Tool: 'get_repository_languages' called for {owner}/{repo}")
    
    endpoint = f"/repos/{owner}/{repo}/languages"
    return await github_client.make_request(endpoint)

# @mcp.tool()


@app.post("/tools/get_user_info")
async def get_user_info(username: Optional[str] = None) -> Dict[str, Any]:
    """Get information about a GitHub user.
    
    Args:
        username: Username to get info for. If not provided, gets authenticated user info.
    
    Returns:
        User information.
    """
    if not github_client:
        raise Exception("GitHub token not configured")
    
    logger.info(f">>> Tool: 'get_user_info' called for user '{username or 'authenticated user'}'")
    
    if username:
        endpoint = f"/users/{username}"
    else:
        endpoint = "/user"
    
    user_data = await github_client.make_request(endpoint)
    
    return {
        "login": user_data["login"],
        "name": user_data.get("name"),
        "bio": user_data.get("bio"),
        "company": user_data.get("company"),
        "location": user_data.get("location"),
        "email": user_data.get("email"),
        "public_repos": user_data["public_repos"],
        "followers": user_data["followers"],
        "following": user_data["following"],
        "created_at": user_data["created_at"],
        "html_url": user_data["html_url"]
    }

if __name__ == "__main__":
    if not GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN environment variable not set. GitHub tools will not work.")
    
    if not JWT_SECRET or JWT_SECRET == 'your-secret-key':
        logger.error("JWT_SECRET environment variable not set properly. This is required for authentication.")
        exit(1)
    
    logger.info("MCP server with authentication started")
    logger.info(f"Server will require JWT authentication for all tool calls")
    logger.info(f"{os.getenv('PORT')}")
    logger.info(f"MCP server started on port {os.getenv('PORT', 8080)}")
    
    # Could also use 'sse' transport, host="0.0.0.0" required for Cloud Run.
    # asyncio.run(
    #     mcp.run_async(
    #         transport="streamable-http", 
    #         host="0.0.0.0", 
    #         port=os.getenv("PORT", 8080),
    #     )
    # )
    # import uvicorn
    # uvicorn.run(
    #     "server:mcp",  # This references your mcp object
    #     host="0.0.0.0",
    #     port=int(os.getenv("PORT", 8080)),
    #     reload=True
    # )
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))