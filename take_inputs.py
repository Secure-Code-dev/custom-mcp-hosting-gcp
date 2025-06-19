import asyncio
import aiohttp
import json

class BridgeClient:
    def __init__(self, bridge_url: str = "http://localhost:8000"):
        self.bridge_url = bridge_url
    
    async def send_message(self, message: str, conversation_id: str = "default", model: str = "chatgpt", model_name: str = None):
        """Send a message to the ChatGPT-Claude-MCP bridge"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "message": message,
                "conversation_id": conversation_id,
                "model": model
            }
            
            if model_name:
                payload["model_name"] = model_name
            
            try:
                async with session.post(f"{self.bridge_url}/chat", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        return {"error": f"HTTP {response.status}: {error_text}"}
            except Exception as e:
                return {"error": f"Connection error: {str(e)}"}
    
    async def list_available_tools(self):
        """List available tools from the bridge"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.bridge_url}/tools") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": f"HTTP {response.status}"}
            except Exception as e:
                return {"error": f"Connection error: {str(e)}"}

async def interactive_chat():
    """Interactive chat interface"""
    client = BridgeClient()
    
    print("🤖 ChatGPT & Claude MCP Bridge Client")
    print("="*50)
    
    # Check available tools
    print("Checking available tools...")
    tools_response = await client.list_available_tools()
    if "error" not in tools_response:
        tools = tools_response.get("tools", [])
        print(f"✅ Connected! Available tools: {len(tools)}")
        for tool in tools:
            print(f"  • {tool['name']}: {tool['description']}")
    else:
        print(f"❌ Error getting tools: {tools_response['error']}")
        return
    
    print("\nAvailable AI Models:")
    print("  • chatgpt (default) - OpenAI GPT models")
    print("  • claude - Anthropic Claude models")
    print("\nStarting chat... (type 'quit' to exit, 'switch' to change model)")
    print("Commands:")
    print("  • /model chatgpt - Switch to ChatGPT")
    print("  • /model claude - Switch to Claude")
    print("  • /help - Show available commands")
    print("-" * 50)
    
    conversation_id = "interactive_session"
    current_model = "chatgpt"
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n🧑 You ({current_model}): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == '/help':
                print("Available commands:")
                print("  • /model chatgpt - Switch to ChatGPT")
                print("  • /model claude - Switch to Claude")
                print("  • /help - Show this help")
                print("  • quit/exit/bye - Exit the chat")
                continue
            
            if user_input.startswith('/model '):
                new_model = user_input[7:].strip().lower()
                if new_model in ['chatgpt', 'claude']:
                    current_model = new_model
                    print(f"🔄 Switched to {current_model}")
                else:
                    print("❌ Invalid model. Use 'chatgpt' or 'claude'")
                continue
            
            if not user_input:
                continue
            
            # Send message to bridge
            print(f"🤖 {current_model.title()}: ", end="", flush=True)
            response = await client.send_message(user_input, conversation_id, current_model)
            
            if "error" in response:
                print(f"❌ Error: {response['error']}")
            else:
                print(response['response'])
                if response.get('tools_used'):
                    print(f"\n🔧 Tools used: {', '.join(response['tools_used'])}")
                if response.get('model_used'):
                    print(f"🧠 Model: {response['model_used']}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

async def test_user_queries():
    """Test queries with user input"""
    client = BridgeClient()
    
    print("🧪 User Query Testing Mode - ChatGPT & Claude")
    print("="*50)
    
    # Check available tools first
    print("Checking available tools...")
    tools_response = await client.list_available_tools()
    if "error" not in tools_response:
        tools = tools_response.get("tools", [])
        print(f"✅ Connected! Available tools: {len(tools)}")
        for tool in tools:
            print(f"  • {tool['name']}: {tool['description']}")
    else:
        print(f"❌ Error getting tools: {tools_response['error']}")
        return
    
    print("\nAvailable AI Models:")
    print("  • chatgpt (default) - OpenAI GPT models")
    print("  • claude - Anthropic Claude models")
    print("\nEnter your queries to test the bridge. Format: [model] your query")
    print("Examples:")
    print("  • chatgpt What's 15 + 27?")
    print("  • claude Get info about GitHub user 'octocat'")
    print("  • What's 100 - 45? (uses default chatgpt)")
    print("Type 'quit' to exit.")
    print("-" * 50)
    
    query_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n🧑 Enter query #{query_count + 1}: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("👋 Exiting test mode!")
                break
            
            if not user_input:
                print("⚠️  Please enter a query or 'quit' to exit.")
                continue
            
            # Parse model and query
            model = "chatgpt"  # default
            query = user_input
            
            # Check if user specified model
            if user_input.startswith("chatgpt "):
                model = "chatgpt"
                query = user_input[8:].strip()
            elif user_input.startswith("claude "):
                model = "claude"
                query = user_input[7:].strip()
            
            query_count += 1
            print(f"\n🔄 Processing query #{query_count} with {model.upper()}: '{query}'")
            print("-" * 40)
            
            # Send message to bridge
            response = await client.send_message(query, f"test_session_{query_count}", model)
            
            if "error" in response:
                print(f"❌ Error: {response['error']}")
            else:
                print(f"🤖 Response: {response['response']}")
                if response.get('tools_used'):
                    print(f"🔧 Tools used: {', '.join(response['tools_used'])}")
                if response.get('model_used'):
                    print(f"🧠 Model: {response['model_used']}")
            
            print(f"✅ Query #{query_count} completed")
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Exiting test mode!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue

async def main():
    """Main function"""
    import sys
    
    print("🤖 ChatGPT-MCP Bridge Test Client")
    print("="*40)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            await interactive_chat()
        elif sys.argv[1] == "test":
            await test_user_queries()
        else:
            print("Usage:")
            print("  python bridge_client_test.py           - User input testing mode")
            print("  python bridge_client_test.py test      - User input testing mode")
            print("  python bridge_client_test.py interactive - Interactive chat mode")
    else:
        # Default to user input testing mode
        await test_user_queries()

if __name__ == "__main__":
    asyncio.run(main())