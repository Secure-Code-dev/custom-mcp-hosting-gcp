import asyncio
import logging
import os
import base64
from typing import List, Dict, Any, Optional
import aiohttp
from fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)

mcp = FastMCP("GitHub MCP Server on Cloud Run")

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

@mcp.tool()
def add(a: int, b: int) -> int:
    """Use this to add two numbers together.
    
    Args:
        a: The first number.
        b: The second number.
    
    Returns:
        The sum of the two numbers.
    """
    logger.info(f">>> Tool: 'add' called with numbers '{a}' and '{b}'")
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Use this to subtract two numbers.
    
    Args:
        a: The first number.
        b: The second number.
    
    Returns:
        The difference of the two numbers.
    """
    logger.info(f">>> Tool: 'subtract' called with numbers '{a}' and '{b}'")
    return a - b

@mcp.tool()
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

@mcp.tool()
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

@mcp.tool()
async def get_file_content(owner: str, repo: str, path: str) -> Dict[str, Any]:
    """Get the content of a specific file from a repository.
    
    Args:
        owner: Repository owner username.
        repo: Repository name.
        path: Path to the file within the repository.
    
    Returns:
        File content and metadata.
    """
    if not github_client:
        raise Exception("GitHub token not configured")
    
    logger.info(f">>> Tool: 'get_file_content' called for {owner}/{repo}/{path}")
    
    endpoint = f"/repos/{owner}/{repo}/contents/{path}"
    file_data = await github_client.make_request(endpoint)
    
    if file_data["type"] != "file":
        raise Exception(f"Path '{path}' is not a file")
    
    # Decode base64 content
    content = base64.b64decode(file_data["content"]).decode('utf-8')
    
    return {
        "name": file_data["name"],
        "path": file_data["path"],
        "size": file_data["size"],
        "content": content,
        "sha": file_data["sha"],
        "encoding": file_data["encoding"]
    }

@mcp.tool()
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

@mcp.tool()
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

@mcp.tool()
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
    
    logger.info(f"MCP server started on port {os.getenv('PORT', 8080)}")
    # Could also use 'sse' transport, host="0.0.0.0" required for Cloud Run.
    asyncio.run(
        mcp.run_async(
            transport="streamable-http", 
            host="0.0.0.0", 
            port=os.getenv("PORT", 8080),
        )
    )