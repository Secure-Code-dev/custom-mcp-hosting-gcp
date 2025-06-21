import asyncio
import aiohttp
import json
import webbrowser
import os
from typing import Optional, Dict, Any

class BridgeClient:
    def __init__(self, bridge_url: str = "http://localhost:8000"):
        self.bridge_url = bridge_url
        self.access_token: Optional[str] = None
        self.user_info: Optional[Dict[str, Any]] = None
        self.session_file = "bridge_session.json"
        self._load_session()
    
    def _save_session(self):
        """Save current session to file"""
        if self.access_token and self.user_info:
            session_data = {
                "access_token": self.access_token,
                "user_info": self.user_info
            }
            try:
                with open(self.session_file, 'w') as f:
                    json.dump(session_data, f)
            except Exception as e:
                print(f"⚠️  Warning: Could not save session: {e}")
    
    def _load_session(self) -> bool:
        """Load saved session from file"""
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)
                    self.access_token = session_data.get("access_token")
                    self.user_info = session_data.get("user_info")
                    return bool(self.access_token and self.user_info)
        except Exception as e:
            print(f"⚠️  Warning: Could not load session: {e}")
        return False
    
    def _clear_session(self):
        """Clear current session"""
        self.access_token = None
        self.user_info = None
        try:
            if os.path.exists(self.session_file):
                os.remove(self.session_file)
        except Exception as e:
            print(f"⚠️  Warning: Could not clear session file: {e}")
    
    async def login(self, provider: str = "github"):
        """Initiate OAuth login flow"""
        print(f"🔐 Starting OAuth login with {provider}...")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Get OAuth login URL
                async with session.get(f"{self.bridge_url}/auth/login/{provider}") as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return {"error": f"Failed to get login URL: HTTP {response.status}: {error_text}"}
                    
                    login_data = await response.json()
                    auth_url = login_data.get("auth_url")
                    
                    if not auth_url:
                        return {"error": "No auth URL received from server"}
                    
                    print(f"🌐 Opening browser for {provider} authentication...")
                    print(f"📝 Auth URL: {auth_url}")
                    
                    # Open browser for authentication
                    webbrowser.open(auth_url)
                    
                    print("\n" + "="*60)
                    print("🔗 Please complete the authentication in your browser")
                    print("📋 After authentication, you'll receive a JSON response")
                    print("🔑 Copy the 'access_token' value from the response")
                    print("="*60)
                    
                    # Get token from user input
                    while True:
                        token_input = input("\n🔑 Paste your access_token here (or 'cancel' to abort): ").strip()
                        
                        if token_input.lower() == 'cancel':
                            return {"error": "Authentication cancelled by user"}
                        
                        if not token_input:
                            print("⚠️  Please enter a valid token or 'cancel'")
                            continue
                        
                        # Verify the token
                        auth_result = await self._verify_token(token_input)
                        if "error" not in auth_result:
                            self.access_token = token_input
                            self.user_info = auth_result.get("user")
                            self._save_session()
                            print(f"✅ Successfully authenticated as {self.user_info.get('name')} ({self.user_info.get('email')})")
                            return {"success": True, "user": self.user_info}
                        else:
                            print(f"❌ Token verification failed: {auth_result['error']}")
                            print("🔄 Please try again with a valid token")
                    
            except Exception as e:
                return {"error": f"Authentication error: {str(e)}"}
    
    async def _verify_token(self, token: str) -> Dict[str, Any]:
        """Verify access token with the server"""
        headers = {"Authorization": f"Bearer {token}"}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.bridge_url}/auth/me", headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        return {"error": f"Token verification failed: HTTP {response.status}: {error_text}"}
            except Exception as e:
                return {"error": f"Token verification error: {str(e)}"}
    
    async def logout(self):
        """Logout and clear session"""
        if not self.access_token:
            print("⚠️  Not currently logged in")
            return
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.bridge_url}/auth/logout", headers=headers) as response:
                    if response.status == 200:
                        print("✅ Successfully logged out")
                    else:
                        print(f"⚠️  Logout request failed: HTTP {response.status}")
            except Exception as e:
                print(f"⚠️  Logout error: {e}")
            finally:
                self._clear_session()
    
    async def check_auth(self) -> bool:
        """Check if currently authenticated"""
        # Try to load saved session first
        if not self.access_token and self._load_session():
            # Verify the loaded token is still valid
            if self.access_token:
                auth_result = await self._verify_token(self.access_token)
                if "error" in auth_result:
                    print("⚠️  Saved session is no longer valid")
                    self._clear_session()
                    return False
                else:
                    self.user_info = auth_result.get("user")
                    return True
        
        return bool(self.access_token)
    
    async def ensure_authenticated(self, provider: str = "github") -> bool:
        """Ensure user is authenticated, prompt for login if not"""
        if await self.check_auth():
            return True
        
        print("🔐 Authentication required")
        print(f"Available providers: github, google")
        
        # Ask user for provider preference
        while True:
            user_provider = input(f"🔑 Choose authentication provider [{provider}]: ").strip().lower()
            if not user_provider:
                user_provider = provider
            
            if user_provider in ['github', 'google']:
                break
            else:
                print("❌ Invalid provider. Please choose 'github' or 'google'")
        
        auth_result = await self.login(user_provider)
        return "error" not in auth_result
    
    async def send_message(self, message: str, conversation_id: str = "default", model: str = "chatgpt", model_name: str = None):
        """Send a message to the ChatGPT-Claude-MCP bridge"""
        # Ensure authentication
        if not await self.ensure_authenticated():
            return {"error": "Authentication failed"}
        
        headers = {"Authorization": f"{self.access_token}"}
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "message": message,
                "conversation_id": conversation_id,
                "model": model,
            }
            
            if model_name:
                payload["model_name"] = model_name
            
            try:
                async with session.post(f"{self.bridge_url}/chat", json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    elif response.status == 401:
                        print("🔐 Authentication expired. Please login again.")
                        self._clear_session()
                        return {"error": "Authentication expired"}
                    else:
                        error_text = await response.text()
                        return {"error": f"HTTP {response.status}: {error_text}"}
            except Exception as e:
                return {"error": f"Connection error: {str(e)}"}
    
    async def list_available_tools(self):
        """List available tools from the bridge"""
        # Ensure authentication
        if not await self.ensure_authenticated():
            return {"error": "Authentication failed"}
                
        headers = {"Authorization": f"Bearer {self.access_token}"}

        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.bridge_url}/tools", headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 401:
                        print("🔐 Authentication expired. Please login again.")
                        self._clear_session()
                        return {"error": "Authentication expired"}
                    else:
                        return {"error": f"HTTP {response.status}"}
            except Exception as e:
                return {"error": f"Connection error: {str(e)}"}

async def interactive_chat():
    """Interactive chat interface with authentication"""
    client = BridgeClient()
    # print("client.access_token ",client.access_token)
    
    print("🤖 ChatGPT & Claude MCP Bridge Client (with OAuth)")
    print("="*50)
    
    # Check authentication
    if await client.check_auth():
        print(f"✅ Already authenticated as {client.user_info.get('name')} ({client.user_info.get('email')})")
    else:
        print("🔐 Authentication required to continue")
        if not await client.ensure_authenticated():
            print("❌ Authentication failed. Exiting.")
            return
    
    # Check available tools
    print("Checking available tools...")
    tools_response = await client.list_available_tools()
    print("tools_response", tools_response)
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
    print("\nStarting chat... (type 'quit' to exit)")
    print("Commands:")
    print("  • /model chatgpt - Switch to ChatGPT")
    print("  • /model claude - Switch to Claude")
    print("  • /logout - Logout and clear session")
    print("  • /whoami - Show current user info")
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
                print("  • /logout - Logout and clear session")
                print("  • /whoami - Show current user info")
                print("  • /help - Show this help")
                print("  • quit/exit/bye - Exit the chat")
                continue
            
            if user_input.lower() == '/logout':
                await client.logout()
                print("🔐 Please authenticate again to continue")
                if not await client.ensure_authenticated():
                    print("❌ Authentication failed. Exiting.")
                    break
                continue
            
            if user_input.lower() == '/whoami':
                if client.user_info:
                    print(f"👤 Current user: {client.user_info.get('name')} ({client.user_info.get('email')})")
                    print(f"🔗 Provider: {client.user_info.get('provider')}")
                else:
                    print("❓ No user information available")
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
                if "Authentication expired" in response['error']:
                    print("🔐 Please authenticate again to continue")
                    if not await client.ensure_authenticated():
                        print("❌ Authentication failed. Exiting.")
                        break
                    continue
                else:
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
    """Test queries with user input and authentication"""
    client = BridgeClient()
    
    print("🧪 User Query Testing Mode - ChatGPT & Claude (with OAuth)")
    print("="*50)
    
    # Check authentication
    if await client.check_auth():
        print(f"✅ Already authenticated as {client.user_info.get('name')} ({client.user_info.get('email')})")
    else:
        print("🔐 Authentication required to continue")
        if not await client.ensure_authenticated():
            print("❌ Authentication failed. Exiting.")
            return
    
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
    print("\nEnter your queries to test the bridge. Format: [model] your query")
    print("Examples:")
    print("  • claude Get info about GitHub user 'octocat'")
    print("Type 'quit' to exit, '/logout' to logout, '/whoami' for user info.")
    print("-" * 50)
    
    query_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n🧑 Enter query #{query_count + 1}: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("👋 Exiting test mode!")
                break
            
            if user_input.lower() == '/logout':
                await client.logout()
                print("🔐 Please authenticate again to continue")
                if not await client.ensure_authenticated():
                    print("❌ Authentication failed. Exiting.")
                    break
                continue
            
            if user_input.lower() == '/whoami':
                if client.user_info:
                    print(f"👤 Current user: {client.user_info.get('name')} ({client.user_info.get('email')})")
                    print(f"🔗 Provider: {client.user_info.get('provider')}")
                else:
                    print("❓ No user information available")
                continue
            
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
                if "Authentication expired" in response['error']:
                    print("🔐 Please authenticate again to continue")
                    if not await client.ensure_authenticated():
                        print("❌ Authentication failed. Exiting.")
                        break
                    continue
                else:
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
    """Main function with authentication options"""
    import sys
    
    print("🤖 ChatGPT-MCP Bridge Client with OAuth Authentication")
    print("="*50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            await interactive_chat()
        elif sys.argv[1] == "test":
            await test_user_queries()
        elif sys.argv[1] == "login":
            # Direct login mode
            client = BridgeClient()
            provider = sys.argv[2] if len(sys.argv) > 2 else "github"
            result = await client.login(provider)
            if "error" not in result:
                print(f"✅ Login successful!")
            else:
                print(f"❌ Login failed: {result['error']}")
        elif sys.argv[1] == "logout":
            # Direct logout mode
            client = BridgeClient()
            await client.logout()
        else:
            print("Usage:")
            print("  python bridge_client_test.py           - User input testing mode")
            print("  python bridge_client_test.py test      - User input testing mode")
            print("  python bridge_client_test.py interactive - Interactive chat mode")
            print("  python bridge_client_test.py login [provider] - Login with OAuth")
            print("  python bridge_client_test.py logout    - Logout and clear session")
    else:
        # Default to user input testing mode
        await test_user_queries()

if __name__ == "__main__":
    asyncio.run(main())