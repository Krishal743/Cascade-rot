"""
HuggingFace Authentication Setup

Instructions:
1. Create a free account at https://huggingface.co (if you don't have one)
2. For Llama-3.1: Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and accept the license
3. Generate a token: https://huggingface.co/settings/tokens (read-only access)
4. Run this script and enter your token when prompted

Qwen and Mistral do NOT require authentication - only needed for Llama.
"""

import os
from pathlib import Path

def setup_huggingface_auth():
    from huggingface_hub import login
    
    token = input("Enter your HuggingFace token: ").strip()
    
    # Login to HuggingFace
    login(token=token)
    
    # Verify login
    from huggingface_hub import whoami
    user_info = whoami()
    print(f"✓ Successfully authenticated as: {user_info['name']}")
    
    # Save token path for reference
    config_dir = Path.home() / ".cache" / "huggingface"
    config_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Token saved to ~/.cache/huggingface/")
    
    return True

if __name__ == "__main__":
    setup_huggingface_auth()
