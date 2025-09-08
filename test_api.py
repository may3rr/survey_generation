#!/usr/bin/env python3
"""
Simple API test script to debug API connectivity issues
"""

import os
import sys
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_direct():
    """Test API with direct requests call"""
    api_key = os.getenv('GPT_API_KEY')
    base_url = os.getenv('GPT_BASE_URL', 'https://api.gpt.ge/v1')
    model = os.getenv('GPT_MODEL', 'gpt-4o-mini')
    
    print("🔍 Direct API Test")
    print(f"API Key: {api_key}")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print("-" * 50)
    
    if not api_key or api_key == 'NOT SET':
        print("❌ API Key not found in environment variables")
        return False
    
    # Test models endpoint first
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    models_url = f"{base_url}/models" if base_url.endswith('/v1') else f"{base_url}/v1/models"
    print(f"Testing models endpoint: {models_url}")
    
    try:
        response = requests.get(models_url, headers=headers, timeout=10)
        print(f"Models endpoint status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Models endpoint working")
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                print(f"Available models: {[m.get('id', 'unknown') for m in data['data'][:3]]}...")
        else:
            print(f"❌ Models endpoint failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False
    
    # Test chat completions endpoint
    chat_url = f"{base_url}/chat/completions" if base_url.endswith('/v1') else f"{base_url}/v1/chat/completions"
    print(f"\nTesting chat endpoint: {chat_url}")
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Hello, this is a test message. Please respond with 'API test successful'."}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(chat_url, headers=headers, json=payload, timeout=30)
        print(f"Chat endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Chat endpoint working")
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                message = data['choices'][0].get('message', {}).get('content', 'No content')
                print(f"API Response: {message}")
                return True
        else:
            print(f"❌ Chat endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat endpoint error: {e}")
        return False

def test_with_project_api_client():
    """Test using the project's API client"""
    print("\n🔍 Project API Client Test")
    print("-" * 50)
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))
    
    try:
        from src.utils.config_loader import load_config
        from src.utils.api_client import GPTAPIClient
        
        # Load config
        config = load_config()
        gpt_config = config.get('api', {}).get('gpt', {})
        
        print(f"Config loaded:")
        print(f"  API Key: {gpt_config.get('api_key', 'NOT FOUND')}")
        print(f"  Base URL: {gpt_config.get('base_url', 'NOT FOUND')}")
        print(f"  Model: {gpt_config.get('model', 'NOT FOUND')}")
        
        # Initialize API client
        client = GPTAPIClient(
            api_key=gpt_config.get('api_key'),
            base_url=gpt_config.get('base_url', 'https://api.gpt.ge/v1'),
            model=gpt_config.get('model', 'gpt-4o-mini')
        )
        
        # Test API call
        test_prompt = "Hello, this is a test message. Please respond with 'Project API client test successful'."
        
        print("\nTesting API call through project client...")
        response = client.generate_text(test_prompt, max_tokens=50)
        
        if response:
            print("✅ Project API client working")
            print(f"Response: {response}")
            return True
        else:
            print("❌ Project API client failed - no response")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Project API client error: {e}")
        return False

def main():
    print("🚀 API Connectivity Test")
    print("=" * 60)
    
    # Test 1: Direct API call
    direct_success = test_api_direct()
    
    # Test 2: Project API client
    project_success = test_with_project_api_client()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"  Direct API call: {'✅ SUCCESS' if direct_success else '❌ FAILED'}")
    print(f"  Project API client: {'✅ SUCCESS' if project_success else '❌ FAILED'}")
    
    if direct_success and project_success:
        print("\n🎉 All tests passed! API is working correctly.")
        return 0
    elif direct_success:
        print("\n⚠️  Direct API works, but project client has issues.")
        return 1
    else:
        print("\n❌ API connectivity issues detected.")
        print("\nPossible solutions:")
        print("1. Check if API key is valid and not expired")
        print("2. Verify base URL is correct")
        print("3. Check if the model name is supported")
        print("4. Test with different API endpoint")
        return 2

if __name__ == "__main__":
    sys.exit(main())