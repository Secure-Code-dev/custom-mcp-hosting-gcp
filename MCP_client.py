import asyncio
import aiohttp
import urllib
import json
import jwt
import time
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from fastmcp import Client
import openai
import anthropic
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OAuth Configuration
OAUTH_PROVIDERS = {
    'github': {
        'client_id': os.getenv('GITHUB_OAUTH_CLIENT_ID'),
        'client_secret': os.getenv('GITHUB_OAUTH_CLIENT_SECRET'),
        'authorize_url': 'https://github.com/login/oauth/authorize',
        'token_url': 'https://github.com/login/oauth/access_token',
        'user_info_url': 'https://api.github.com/user',
        'scope': 'user:email repo'
    },
    'google': {
        'client_id': os.getenv('GOOGLE_OAUTH_CLIENT_ID'),
        'client_secret': os.getenv('GOOGLE_OAUTH_CLIENT_SECRET'),
        'authorize_url': 'https://accounts.google.com/o/oauth2/auth',
        'token_url': 'https://oauth2.googleapis.com/token',
        'user_info_url': 'https://www.googleapis.com/oauth2/v2/userinfo',
        'scope': 'openid email profile'
    }
}

# JWT Configuration
JWT_SECRET = os.getenv('JWT_SECRET', secrets.token_urlsafe(32))
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))

# Server Configuration
SERVER_URL = os.getenv('CLIENT_SERVER_URL', 'http://localhost:8000')
REDIRECT_URI = f"{SERVER_URL}/auth/callback"

class AuthenticationManager:
    def __init__(self):
        self.active_sessions = {}  # In production, use Redis or database
        self.oauth_states = {}     # Store OAuth state tokens
    
    def generate_jwt_token(self, user_data: Dict[str, Any]) -> str:
        """Generate JWT token for authenticated user"""
        payload = {
            'user_id': user_data.get('id'),
            'email': user_data.get('email'),
            'name': user_data.get('name'),
            'provider': user_data.get('provider'),
            'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
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
    
    def generate_oauth_state(self) -> str:
        """Generate secure state parameter for OAuth"""
        state = secrets.token_urlsafe(32)
        self.oauth_states[state] = {
            'created_at': time.time(),
            'used': False
        }
        return state
    
    def verify_oauth_state(self, state: str) -> bool:
        """Verify OAuth state parameter"""
        if state not in self.oauth_states:
            return False
        
        state_data = self.oauth_states[state]
        
        # Check if state is expired (5 minutes)
        if time.time() - state_data['created_at'] > 300:
            del self.oauth_states[state]
            return False
        
        # Check if state is already used
        if state_data['used']:
            return False
        
        # Mark as used
        state_data['used'] = True
        return True

auth_manager = AuthenticationManager()
security = HTTPBearer()

class OAuthClient:
    def __init__(self, provider: str):
        self.provider = provider
        self.config = OAUTH_PROVIDERS.get(provider)
        if not self.config:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
        
        if not self.config['client_id'] or not self.config['client_secret']:
            raise ValueError(f"OAuth credentials not configured for {provider}")
    
    def get_authorization_url(self, state: str) -> str:
        """Generate OAuth authorization URL"""
        
        
        if self.provider == 'google':
            params = {
                'client_id': self.config['client_id'],
                'redirect_uri': f"{REDIRECT_URI}/google",
                'scope': self.config['scope'],
                'state': state,
                'response_type': 'code'
            }
            params['access_type'] = 'offline'
        else:
            params = {
                'client_id': self.config['client_id'],
                'redirect_uri': REDIRECT_URI,
                'scope': self.config['scope'],
                'state': state,
                'response_type': 'code'
            }
            
        
        return f"{self.config['authorize_url']}?{urllib.parse.urlencode(params)}"
    
    async def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        if self.provider == 'google':

            data = {
                'client_id': self.config['client_id'],
                'client_secret': self.config['client_secret'],
                'code': code,
                'redirect_uri': f"{REDIRECT_URI}/google",
                'grant_type': 'authorization_code'
            }
        else:
            data = {
                'client_id': self.config['client_id'],
                'client_secret': self.config['client_secret'],
                'code': code,
                'redirect_uri': REDIRECT_URI,
                'grant_type': 'authorization_code'
            }
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config['token_url'], 
                data=data, 
                headers=headers
            ) as response:
                
                r = await response.json()
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to exchange code for token: {error_text}"
                    )
                return await response.json()
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information using access token"""
        headers = {
            'Authorization': f'Bearer {str(access_token)}',
            'Accept': 'application/json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.config['user_info_url'], 
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to get user info: {error_text}"
                    )
                
                user_data = await response.json()
                user_data['provider'] = self.provider
                return user_data

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Dependency to get current authenticated user"""
    try:
        token = credentials.credentials
        print("token from get_current_user", token)
        user_data = auth_manager.verify_jwt_token(token)
        print("user_data from get_current_user", user_data)
        return user_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    conversation_id: str = "default"
    model: str = "chatgpt"  # "chatgpt" or "claude"
    model_name: Optional[str] = None  # Specific model name

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    tools_used: List[str] = []
    model_used: str

class AuthenticatedMCPClient:
    """MCP Client with JWT authentication"""
    
    def __init__(self, mcp_server_url: str, jwt_token: str = None):
        self.mcp_server_url = mcp_server_url
        self.jwt_token = jwt_token
        self.headers = {}
        if jwt_token:
            self.headers['Authorization'] = f'Bearer {jwt_token}'
    
    def set_jwt_token(self, token: str):
        """Set JWT token for authentication"""
        self.jwt_token = token
        self.headers['Authorization'] = f'Bearer {token}'
    
    async def create_tools(self):
        """List available tools from MCP server with authentication"""

        try:
            # async with Client(self.mcp_server_url) as client:

                # print("printing client now",client)
            self.available_tools = [
                {
                    "name": "add",
                    "description": "Use this to add two numbers together.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "a": {"title": "A", "type": "integer"},
                            "b": {"title": "B", "type": "integer"}
                        },
                        "required": ["a", "b"]
                    },
                    "annotations": None
                },
                {
                    "name": "subtract",
                    "description": "Use this to subtract two numbers.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "a": {"title": "A", "type": "integer"},
                            "b": {"title": "B", "type": "integer"}
                        },
                        "required": ["a", "b"]
                    },
                    "annotations": None
                },
                {
                    "name": "list_repositories",
                    "description": "List repositories for the authenticated user or a specific username.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "title": "Username"
                            }
                        }
                    },
                    "annotations": None
                },
                {
                    "name": "get_repository_contents",
                    "description": "Get contents of a repository directory or file.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "owner": {"title": "Owner", "type": "string"},
                            "repo": {"title": "Repo", "type": "string"},
                            "path": {"default": "", "title": "Path", "type": "string"}
                        },
                        "required": ["owner", "repo"]
                    },
                    "annotations": None
                },
                {
                    "name": "get_file_content",
                    "description": "Get the content of a specific file from a repository.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "owner": {"title": "Owner", "type": "string"},
                            "repo": {"title": "Repo", "type": "string"},
                            "path": {"title": "Path", "type": "string"}
                        },
                        "required": ["owner", "repo", "path"]
                    },
                    "annotations": None
                },
                {
                    "name": "search_code",
                    "description": "Search for code across repositories.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"title": "Query", "type": "string"},
                            "owner": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "title": "Owner"
                            },
                            "repo": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "title": "Repo"
                            }
                        },
                        "required": ["query"]
                    },
                    "annotations": None
                },
                {
                    "name": "get_repository_languages",
                    "description": "Get programming languages used in a repository.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "owner": {"title": "Owner", "type": "string"},
                            "repo": {"title": "Repo", "type": "string"}
                        },
                        "required": ["owner", "repo"]
                    },
                    "annotations": None
                },
                {
                    "name": "get_user_info",
                    "description": "Get information about a GitHub user.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "title": "Username"
                            }
                        }
                    },
                    "annotations": None
                }
            ]

        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            raise
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> List[Any]:
        """Call a tool on the MCP server with authentication"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.mcp_server_url}/tools/{tool_name}",
                    json=parameters,
                    # headers={
                    #     **self.headers,
                    #     'Content-Type': 'application/json'
                    # }
                ) as response:
                    if response.status == 401:
                        raise HTTPException(
                            status_code=401,
                            detail="Authentication required or token expired"
                        )
                    elif response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Tool call failed: {error_text}")
                    
                    result = await response.json()
                    # Return result in format expected by existing code
                    # return [type('Result', (), {'text': result.get('result', '')})()]
                    return result
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            raise

class MCPBridge:
    def __init__(self, mcp_server_url: str = "http://container1:8080"):
        self.mcp_server_url = mcp_server_url
        self.mcp_client = None
        
        # Initialize OpenAI client
        # try:
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # except TypeError as e:
        #     if "proxies" in str(e):
        #         print("going in proxies")
        #         self.openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        #     else:
        #         raise e
        
        # Initialize Anthropic client
        self.anthropic_client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        self.available_tools = []
        
    async def initialize_with_token(self, jwt_token: str):
        """Initialize the MCP connection with JWT token and get available tools"""
        try:
            self.mcp_client = AuthenticatedMCPClient(self.mcp_server_url, jwt_token)

            # self.available_tools = await self.mcp_client.create_tools()
            self.available_tools = [
                {
                    "name": "add",
                    "description": "Use this to add two numbers together.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "a": {"title": "A", "type": "integer"},
                            "b": {"title": "B", "type": "integer"}
                        },
                        "required": ["a", "b"]
                    },
                    "annotations": None
                },
                {
                    "name": "subtract",
                    "description": "Use this to subtract two numbers.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "a": {"title": "A", "type": "integer"},
                            "b": {"title": "B", "type": "integer"}
                        },
                        "required": ["a", "b"]
                    },
                    "annotations": None
                },
                {
                    "name": "list_repositories",
                    "description": "List repositories for the authenticated user or a specific username.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "title": "Username"
                            }
                        }
                    },
                    "annotations": None
                },
                {
                    "name": "get_repository_contents",
                    "description": "Get contents of a repository directory or file.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "owner": {"title": "Owner", "type": "string"},
                            "repo": {"title": "Repo", "type": "string"},
                            "path": {"default": "", "title": "Path", "type": "string"}
                        },
                        "required": ["owner", "repo"]
                    },
                    "annotations": None
                },
                {
                    "name": "get_file_content",
                    "description": "Get the content of a specific file from a repository.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "owner": {"title": "Owner", "type": "string"},
                            "repo": {"title": "Repo", "type": "string"},
                            "path": {"title": "Path", "type": "string"}
                        },
                        "required": ["owner", "repo", "path"]
                    },
                    "annotations": None
                },
                {
                    "name": "search_code",
                    "description": "Search for code across repositories.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"title": "Query", "type": "string"},
                            "owner": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "title": "Owner"
                            },
                            "repo": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "title": "Repo"
                            }
                        },
                        "required": ["query"]
                    },
                    "annotations": None
                },
                {
                    "name": "get_repository_languages",
                    "description": "Get programming languages used in a repository.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "owner": {"title": "Owner", "type": "string"},
                            "repo": {"title": "Repo", "type": "string"}
                        },
                        "required": ["owner", "repo"]
                    },
                    "annotations": None
                },
                {
                    "name": "get_user_info",
                    "description": "Get information about a GitHub user.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "username": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "title": "Username"
                            }
                        }
                    },
                    "annotations": None
                }
            ]
            # print("available tools", self.available_tools)
            logger.info(f"Connected to MCP server")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def initialize(self):
        """Initialize without token - for backwards compatibility"""
        logger.warning("Initializing MCP bridge without authentication token")
        self.available_tools = []

    def get_tools_for_openai(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI function calling format"""
        openai_tools = []

        for tool in self.available_tools:

            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.get('name'),
                    "description": tool.get('description'),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # Add parameters if available
            if 'inputSchema' in tool and tool['inputSchema']:
                if 'properties' in tool['inputSchema']:
                    tool_def["function"]["parameters"]["properties"] = tool['inputSchema']['properties']
                if 'required' in tool['inputSchema']:
                    tool_def["function"]["parameters"]["required"] = tool['inputSchema']['required']
            
            openai_tools.append(tool_def)
        
        return openai_tools

    def get_tools_for_claude(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to Claude (Anthropic) tool format"""
        claude_tools = []
        
        for tool in self.available_tools:
            # Create Claude tool definition
            tool_def = {
                "name": tool.get('name'),
                "description": tool.get('description'),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Add parameters if available
            if 'inputSchema' in tool and tool['inputSchema']:
                if 'properties' in tool['inputSchema']:
                    tool_def["input_schema"]["properties"] = tool['inputSchema']['properties']
                if 'required' in tool['inputSchema']:
                    tool_def["input_schema"]["required"] = tool['inputSchema']['required']
            
            claude_tools.append(tool_def)
        
        return claude_tools

    async def call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Call an MCP tool and return the result"""
        try:
            if not self.mcp_client:
                raise Exception("MCP client not initialized with authentication")
            
            result = await self.mcp_client.call_tool(tool_name, parameters)
            # return result[0].text if result else "No result returned"
            return result
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    async def process_chat_message_claude(self, message: str, conversation_id: str, model_name: str = "claude-3-5-sonnet-20241022", jwt_token: str = None) -> ChatResponse:
        """Process a chat message using Claude with MCP tool integration"""
        tools_used = []
        
        # Initialize MCP client with token if provided
        if jwt_token and not self.mcp_client:
            await self.initialize_with_token(jwt_token)
        
        try:
            # Get Claude-formatted tools
            claude_tools = self.get_tools_for_claude()
            
            # Create system message explaining available tools
            system_message = f"""You are an AI assistant with access to various tools through an MCP (Model Context Protocol) server.

Available tools:
{chr(10).join([f"- {tool.get('name')}: {tool.get('description')}" for tool in self.available_tools])}

When a user asks something that could benefit from using these tools, use the appropriate tool calls. 
For example:
- For math operations, use add/subtract tools
- For GitHub-related queries, use the GitHub tools (get_user_info, list_repositories, etc.)
- Always provide helpful context and explanations with your responses

If you need to use multiple tools or chain operations, do so to provide the most complete answer possible."""

            # Call Claude with tools
            response = await self.anthropic_client.messages.create(
                model=model_name,
                max_tokens=2500,
                system=system_message,
                messages=[
                    {"role": "user", "content": message}
                ],
                tools=claude_tools if claude_tools else None
            )

            # Process the response
            response_content = ""
            tool_calls = []
            
            for content_block in response.content:
                if content_block.type == "text":
                    response_content += content_block.text
                elif content_block.type == "tool_use":
                    tool_calls.append(content_block)

            # Handle tool calls if any
            if tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call.name
                    tool_args = tool_call.input
                    
                    logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
                    tools_used.append(tool_name)
                    
                    # Call the MCP tool
                    result = await self.call_mcp_tool(tool_name, tool_args)
                    
                    if tool_name == "get_file_content":
    
                        tool_results.append({
                            "tool_use_id": tool_call.id,
                            "type": "tool_result",
                            "content": result["content"]
                        })
                    elif tool_name == "get_repository_contents":
                        print("get_repository_contents block")
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": result
                        })
                    else:
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": result
                        })
                    

                # Get final response with tool results
                messages = [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": tool_results}
                ]


                final_response = await self.anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=2500,
                    system=system_message,
                    messages=messages
                )
                
                final_content = ""
                for content_block in final_response.content:
                    if content_block.type == "text":
                        final_content += content_block.text
            else:
                final_content = response_content

            return ChatResponse(
                response=final_content or "I'm sorry, I couldn't generate a response.",
                conversation_id=conversation_id,
                tools_used=tools_used,
                model_used=f"claude-{model_name}"
            )

        except Exception as e:
            logger.error(f"Error processing chat message with Claude: {e}")
            return ChatResponse(
                response=f"I encountered an error while processing your request: {str(e)}",
                conversation_id=conversation_id,
                tools_used=tools_used,
                model_used=f"claude-{model_name}"
            )

    async def process_chat_message(self, message: str, conversation_id: str, model: str = "chatgpt", model_name: str = None, jwt_token: str = None) -> ChatResponse:
        """Process a chat message using the specified AI model"""
        if model.lower() == "claude":
            claude_model = model_name or "claude-3-5-sonnet-20241022"
            return await self.process_chat_message_claude(message, conversation_id, claude_model, jwt_token)
        else:
            # Default to ChatGPT
            return await self.process_chat_message_chatgpt(message, conversation_id, model_name or "gpt-4o-mini", jwt_token)

    async def process_chat_message_chatgpt(self, message: str, conversation_id: str, model_name: str = "gpt-4o-mini", jwt_token: str = None) -> ChatResponse:
        """Process a chat message using ChatGPT with MCP tool integration"""
        tools_used = []
        
        # Initialize MCP client with token if provided
        if jwt_token and not self.mcp_client:
            await self.initialize_with_token(jwt_token)
        
        try:
            # Get OpenAI-formatted tools
            openai_tools = self.get_tools_for_openai()
            
            # Create system message explaining available tools
            system_message = f"""You are an AI assistant with access to various tools through an MCP (Model Context Protocol) server.

Available tools:
{chr(10).join([f"- {tool.get('name')}: {tool.get('description')}" for tool in self.available_tools])}

When a user asks something that could benefit from using these tools, use the appropriate function calls. 
For example:
- For math operations, use add/subtract tools
- For GitHub-related queries, use the GitHub tools (get_user_info, list_repositories, etc.)
- Always provide helpful context and explanations with your responses

If you need to use multiple tools or chain operations, do so to provide the most complete answer possible."""

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message}
            ]

            # Call ChatGPT with tools
            response = await self.openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None,
            )

            assistant_message = response.choices[0].message
            
            # Handle tool calls if any
            if assistant_message.tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    print("tool_name", tool_name)
                    print("tool_args", tool_args)
                    
                    logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
                    tools_used.append(tool_name)
                    
                    # Call the MCP tool
                    result = await self.call_mcp_tool(tool_name, tool_args)

                    if tool_name == "get_file_content":

                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": result["name"],
                            "content": result["content"]
                        })
                    elif tool_name == "get_repository_contents":
                        print("get_repository_contents block")
                        tool_content = "Here are the files in the repo:\n" + "\n".join(
                            f"- {item['name']}" for item in result
                        )
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": tool_content
                        })
                    else:
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": result
                        })

                print("assistant_message.content ", assistant_message.content)

                # Add tool results to conversation and get final response
                messages.append({
                    "role": "assistant", 
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in assistant_message.tool_calls
                    ]
                })
                # print("messages ", messages)
                messages.extend(tool_results)
                print("messages ", messages)

                # print("tool_results ", tool_results)
                # Get final response with tool results
                final_response = await self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
                # print("final response", final_response)

                
                final_content = final_response.choices[0].message.content
                # print("final content", final_content)
            else:
                final_content = assistant_message.content

            return ChatResponse(
                response=final_content or "I'm sorry, I couldn't generate a response.",
                conversation_id=conversation_id,
                tools_used=tools_used,
                model_used=f"chatgpt-{model_name}"
            )

        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return ChatResponse(
                response=f"I encountered an error while processing your request: {str(e)}",
                conversation_id=conversation_id,
                tools_used=tools_used,
                model_used=f"chatgpt-{model_name}"
            )

# Global bridge instance
bridge = MCPBridge()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     await bridge.initialize()
#     yield
#     # Shutdown
#     pass

# FastAPI app
app = FastAPI(title="ChatGPT-MCP Bridge")

@app.get("/auth/login/{provider}")
async def login(provider: str):
    """Initiate OAuth login"""
    if provider not in OAUTH_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported provider: {provider}"
        )
    
    try:
        oauth_client = OAuthClient(provider)
        state = auth_manager.generate_oauth_state()
        auth_url = oauth_client.get_authorization_url(state)
        
        return JSONResponse({
            "auth_url": auth_url,
            "state": state,
            "message": f"Please visit the auth_url to authenticate with {provider}"
        })
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/auth/callback")
async def oauth_callback(code: str, state: str, provider: str = "github"):
    """Handle OAuth callback"""
    # Verify state parameter
    if not auth_manager.verify_oauth_state(state):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired state parameter"
        )
    
    try:
        oauth_client = OAuthClient(provider)        
        # Exchange code for token
        token_data = await oauth_client.exchange_code_for_token(code)

        access_token = token_data.get('access_token')
        access_token = str(access_token)
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to obtain access token"
            )
        
        # Get user information
        user_info = await oauth_client.get_user_info(access_token)
        # Generate JWT token
        jwt_token = auth_manager.generate_jwt_token(user_info)
        
        logger.info(f"User authenticated successfully: {user_info.get('email')}, {user_info.get('login')}")
        
        return JSONResponse({
            "access_token": str(jwt_token),
            "token_type": "bearer",
            "expires_in": JWT_EXPIRATION_HOURS * 3600,
            "user": {
                "id": user_info.get('id'),
                "name": user_info.get('name', user_info.get('login')),
                "email": user_info.get('email'),
                "provider": provider
            }
        })
    
    except Exception as e:
        logger.error(f"OAuth callback error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )
    
@app.get("/auth/callback/google")
async def oauth_callback_google(code: str, state: str, provider: str = "google"):
    """Handle OAuth callback for Google"""
    # Verify state parameter
    if not auth_manager.verify_oauth_state(state):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired state parameter"
        )
    
    try:
        oauth_client = OAuthClient(provider)        
        # Exchange code for token
        token_data = await oauth_client.exchange_code_for_token(code)

        access_token = token_data.get('access_token')
        
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to obtain access token"
            )
        
        # Get user information
        user_info = await oauth_client.get_user_info(access_token)
        
        # Generate JWT token
        jwt_token = auth_manager.generate_jwt_token(user_info)
        
        logger.info(f"User authenticated successfully: {user_info.get('email', user_info.get('login'))}")
        
        return JSONResponse({
            "access_token": str(jwt_token),
            "token_type": "bearer",
            "expires_in": JWT_EXPIRATION_HOURS * 3600,
            "user": {
                "id": user_info.get('id'),
                "name": user_info.get('name', user_info.get('login')),
                "email": user_info.get('email'),
                "provider": provider
            }
        })
    
    except Exception as e:
        logger.error(f"OAuth callback error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )

@app.get("/auth/me")
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get current authenticated user information"""
    return {
        "user": {
            "id": current_user.get('user_id'),
            "name": current_user.get('name'),
            "email": current_user.get('email'),
            "provider": current_user.get('provider')
        },
        "expires_at": current_user.get('exp')
    }

@app.post("/auth/logout")
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logout current user"""
    # In a real implementation, you might want to blacklist the token
    return {"message": "Successfully logged out"}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request_body: ChatRequest, request: Request):

    auth_header = request.headers.get("Authorization")

    # Process message
    return await bridge.process_chat_message(
        message=request_body.message,
        conversation_id=request_body.conversation_id,
        model=request_body.model,
        model_name=request_body.model_name,
        jwt_token=auth_header
    )

@app.get("/tools")
async def list_tools():
    """List available MCP tools"""
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description
            } for tool in bridge.available_tools
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "mcp_server": bridge.mcp_server_url}

# Test endpoint for direct tool calls
@app.post("/tool/{tool_name}")
async def call_tool_directly(tool_name: str, parameters: Dict[str, Any]):
    """Direct tool call endpoint for testing"""
    try:
        result = await bridge.call_mcp_tool(tool_name, parameters)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/")
async def root():
    """Root endpoint with authentication information"""
    return {
        "message": "GitHub MCP Server with OAuth Authentication",
        "version": "1.0.0",
        "authentication": {
            "required": True,
            "providers": list(OAUTH_PROVIDERS.keys()),
            "login_endpoints": {
                provider: f"/auth/login/{provider}" 
                for provider in OAUTH_PROVIDERS.keys()
            }
        },
        "endpoints": {
            "auth": {
                "login": "/auth/login/{provider}",
                "callback": "/auth/callback",
                "user_info": "/auth/me",
                "logout": "/auth/logout"
            },
            "tools": {
                "add": "/tools/add",
                "subtract": "/tools/subtract",
                "list_repositories": "/tools/list_repositories",
                "get_repository_contents": "/tools/get_repository_contents"
            },
            "system": {
                "health": "/health",
                "docs": "/docs"
            }
        }
    }


if __name__ == "__main__":
    # Validate OAuth configuration
    configured_providers = []
    for provider, config in OAUTH_PROVIDERS.items():
        if config['client_id'] and config['client_secret']:
            configured_providers.append(provider)
        else:
            logger.warning(f"OAuth provider '{provider}' not configured - missing credentials")
    
    if not configured_providers:
        logger.error("No OAuth providers configured! Please set OAuth credentials in environment variables.")
        exit(1)
    
    logger.info(f"OAuth providers configured: {configured_providers}")
    logger.info(f"Server URL: {SERVER_URL}")
    logger.info(f"Redirect URI: {REDIRECT_URI}")
    
    logger.info(f"Server starting on port {os.getenv('CLIENT_PORT', 8080)}")
    
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("CLIENT_PORT", 8000)),
        log_level="info"
    )