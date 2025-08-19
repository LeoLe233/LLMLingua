# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from time import sleep
import os
from dotenv import load_dotenv

import openai
import tiktoken

# Load environment variables from .env file
load_dotenv()

def query_llm(
    prompt,
    model,
    model_name,
    max_tokens,
    tokenizer=None,
    chat_completion=False,
    **kwargs,
):
    SLEEP_TIME_FAILED = 62

    request = {
        "temperature": kwargs["temperature"] if "temperature" in kwargs else 0.0,
        "top_p": kwargs["top_p"] if "top_p" in kwargs else 1.0,
        "seed": kwargs["seed"] if "seed" in kwargs else 42,
        "max_tokens": max_tokens,
        "n": 1,
        "stream": False,
    }
    if chat_completion:
        request["messages"] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    else:
        request["prompt"] = prompt

    answer = None
    response = None
    while answer is None:
        try:
            response = model.create(engine=model_name, **request)
            answer = (
                response["choices"][0]["message"]["content"]
                if chat_completion
                else response["choices"][0]["text"]
            )
        except Exception as e:
            answer = None
            print(f"error: {e}, response: {response}")
            sleep(SLEEP_TIME_FAILED)
    # sleep(SLEEP_TIME_SUCCESS)
    return answer


def load_model_and_tokenizer(model_name_or_path, chat_completion=False):
    # Get API credentials from environment variables
    openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai.api_base = os.getenv("AZURE_OPENAI_API_BASE")
    openai.api_type = "azure"
    openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    # Validate that required environment variables are set
    if not openai.api_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set")
    if not openai.api_base:
        raise ValueError("AZURE_OPENAI_API_BASE environment variable is not set")

    if chat_completion:
        model = openai.ChatCompletion
    else:
        model = openai.Completion

    tokenizer = tiktoken.encoding_for_model("gpt-4")
    return model, tokenizer


def test_azure_api_connectivity():
    """
    Test function to check Azure OpenAI API connectivity.
    Returns a tuple of (success: bool, message: str)
    """
    try:
        # Load the model and tokenizer
        model, tokenizer = load_model_and_tokenizer("gpt-35-turbo-0125", chat_completion=True)
        
        # Test with a simple prompt
        test_prompt = "Hello, this is a connectivity test. Please respond with 'Connection successful!'"
        
        # Make a test API call
        response = model.create(
            engine="gpt-35-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": test_prompt}
            ],
            max_tokens=50,
            temperature=0.0,
            top_p=1.0,
            seed=42,
            n=1,
            stream=False
        )
        
        # Check if we got a valid response
        if response and "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            return True, f"✅ Connection successful! Response: {content}"
        else:
            return False, "❌ API call succeeded but response format is unexpected"
            
    except Exception as e:
        return False, f"❌ Connection failed with error: {str(e)}"


def test_azure_api_connectivity_simple():
    """
    Simple connectivity test that just checks if the API can be reached.
    Returns a tuple of (success: bool, message: str)
    """
    try:
        # Load the model and tokenizer
        model, tokenizer = load_model_and_tokenizer("gpt-35-turbo-0125", chat_completion=True)
        
        # Just test if we can create the model object (this tests the configuration)
        if model and tokenizer:
            return True, "✅ API configuration loaded successfully"
        else:
            return False, "❌ Failed to load model or tokenizer"
            
    except Exception as e:
        return False, f"❌ Configuration error: {str(e)}"


# Example usage and testing
if __name__ == "__main__":
    print("Testing Azure OpenAI API connectivity...")
    print("-" * 50)
    
    # Test 1: Simple configuration test
    success, message = test_azure_api_connectivity_simple()
    print(f"Configuration Test: {message}")
    
    # Test 2: Full API connectivity test
    if success:
        print("\nTesting full API connectivity...")
        success, message = test_azure_api_connectivity()
        print(f"API Test: {message}")
    
    print("-" * 50)

