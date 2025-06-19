import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastmcp import Client
import openai
import anthropic
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class MCPBridge:
    def __init__(self, mcp_server_url: str = "http://localhost:8080/mcp"):
        self.mcp_server_url = mcp_server_url
        
        # Initialize OpenAI client
        try:
            self.openai_client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=60.0
            )
        except TypeError as e:
            if "proxies" in str(e):
                self.openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            else:
                raise e
        
        # Initialize Anthropic client
        self.anthropic_client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        self.available_tools = []
        
    async def initialize(self):
        """Initialize the MCP connection and get available tools"""
        try:
            async with Client(self.mcp_server_url) as client:
                self.available_tools = await client.list_tools()
                logger.info(f"Connected to MCP server. Available tools: {[tool.name for tool in self.available_tools]}")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise

    def get_tools_for_openai(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to OpenAI function calling format"""
        openai_tools = []
        
        for tool in self.available_tools:
            # Create OpenAI tool definition
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # Add parameters if available
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                if 'properties' in tool.inputSchema:
                    tool_def["function"]["parameters"]["properties"] = tool.inputSchema['properties']
                if 'required' in tool.inputSchema:
                    tool_def["function"]["parameters"]["required"] = tool.inputSchema['required']
            
            openai_tools.append(tool_def)
        
        return openai_tools

    def get_tools_for_claude(self) -> List[Dict[str, Any]]:
        """Convert MCP tools to Claude (Anthropic) tool format"""
        claude_tools = []
        
        for tool in self.available_tools:
            # Create Claude tool definition
            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Add parameters if available
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                if 'properties' in tool.inputSchema:
                    tool_def["input_schema"]["properties"] = tool.inputSchema['properties']
                if 'required' in tool.inputSchema:
                    tool_def["input_schema"]["required"] = tool.inputSchema['required']
            
            claude_tools.append(tool_def)
        
        return claude_tools
        """Convert MCP tools to OpenAI function calling format"""
        openai_tools = []
        
        for tool in self.available_tools:
            # Create OpenAI tool definition
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # Add parameters if available
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                if 'properties' in tool.inputSchema:
                    tool_def["function"]["parameters"]["properties"] = tool.inputSchema['properties']
                if 'required' in tool.inputSchema:
                    tool_def["function"]["parameters"]["required"] = tool.inputSchema['required']
            
            openai_tools.append(tool_def)
        
        return openai_tools

    async def call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Call an MCP tool and return the result"""
        try:
            async with Client(self.mcp_server_url) as client:
                result = await client.call_tool(tool_name, parameters)
                return result[0].text if result else "No result returned"
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"

    async def process_chat_message_claude(self, message: str, conversation_id: str, model_name: str = "claude-3-5-sonnet-20241022") -> ChatResponse:
        """Process a chat message using Claude with MCP tool integration"""
        tools_used = []
        
        try:
            # Get Claude-formatted tools
            claude_tools = self.get_tools_for_claude()
            
            # Create system message explaining available tools
            system_message = f"""You are an AI assistant with access to various tools through an MCP (Model Context Protocol) server.

Available tools:
{chr(10).join([f"- {tool.name}: {tool.description}" for tool in self.available_tools])}

When a user asks something that could benefit from using these tools, use the appropriate tool calls. 
For example:
- For math operations, use add/subtract tools
- For GitHub-related queries, use the GitHub tools (get_user_info, list_repositories, etc.)
- Always provide helpful context and explanations with your responses

If you need to use multiple tools or chain operations, do so to provide the most complete answer possible."""

            # Call Claude with tools
            response = await self.anthropic_client.messages.create(
                model=model_name,
                max_tokens=1500,
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
                    tool_results.append({
                        "tool_use_id": tool_call.id,
                        "type": "tool_result",
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
                    max_tokens=1500,
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

    async def process_chat_message(self, message: str, conversation_id: str, model: str = "chatgpt", model_name: str = None) -> ChatResponse:
        """Process a chat message using the specified AI model"""
        if model.lower() == "claude":
            claude_model = model_name or "claude-3-5-sonnet-20241022"
            return await self.process_chat_message_claude(message, conversation_id, claude_model)
        else:
            # Default to ChatGPT
            return await self.process_chat_message_chatgpt(message, conversation_id, model_name or "gpt-4")

    async def process_chat_message_chatgpt(self, message: str, conversation_id: str, model_name: str = "gpt-4o-mini") -> ChatResponse:
        """Process a chat message using ChatGPT with MCP tool integration"""
        tools_used = []
        
        try:
            # Get OpenAI-formatted tools
            openai_tools = self.get_tools_for_openai()
            
            # Create system message explaining available tools
            system_message = f"""You are an AI assistant with access to various tools through an MCP (Model Context Protocol) server.

Available tools:
{chr(10).join([f"- {tool.name}: {tool.description}" for tool in self.available_tools])}

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
                max_tokens=1500
            )

            # Process the response
            assistant_message = response.choices[0].message
            
            # Handle tool calls if any
            if assistant_message.tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
                    tools_used.append(tool_name)
                    
                    # Call the MCP tool
                    result = await self.call_mcp_tool(tool_name, tool_args)
                    tool_results.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": result
                    })

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
                
                messages.extend(tool_results)

                # Get final response with tool results
                final_response = await self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=1500
                )
                
                final_content = final_response.choices[0].message.content
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await bridge.initialize()
    yield
    # Shutdown
    pass

# FastAPI app
app = FastAPI(title="ChatGPT-MCP Bridge", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    return await bridge.process_chat_message(
        request.message, 
        request.conversation_id, 
        request.model, 
        request.model_name
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)