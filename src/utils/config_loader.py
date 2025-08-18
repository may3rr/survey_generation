"""
Secure configuration loader for survey generation system.
Loads configuration from YAML files and environment variables.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML files and environment variables.
    Environment variables take precedence over YAML configuration.
    """
    
    # Load .env file
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    config = {}
    
    # Load base configuration from YAML
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    
    # Override with environment variables
    env_mappings = {
        'GPT_API_KEY': ['api', 'gpt', 'api_key'],
        'GPT_BASE_URL': ['api', 'gpt', 'base_url'],
        'GPT_MODEL': ['api', 'gpt', 'model'],
        'QWEN_API_KEY': ['api', 'qwen', 'api_key'],
        'QWEN_BASE_URL': ['api', 'qwen', 'base_url'],
        'QWEN_MODEL': ['api', 'qwen', 'model'],
        'RAW_DATA_PATH': ['data', 'raw_data_path'],
        'PROCESSED_DATA_DIR': ['data', 'processed_data_dir'],
        'OUTPUT_DIR': ['data', 'output_dir'],
        'DEFAULT_NUM_SURVEYS': ['generation', 'num_surveys'],
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value:
            # Convert string to int for numeric values
            if env_var == 'DEFAULT_NUM_SURVEYS':
                try:
                    value = int(value)
                except ValueError:
                    continue
            
            # Set nested configuration
            current = config
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[config_path[-1]] = value
    
    # Ensure required directories exist
    os.makedirs(config.get('data', {}).get('processed_data_dir', './data/processed'), exist_ok=True)
    os.makedirs(config.get('data', {}).get('output_dir', './outputs'), exist_ok=True)
    
    return config

def get_api_config(config: Dict[str, Any], api_type: str = 'gpt') -> Dict[str, Any]:
    """Get API-specific configuration"""
    api_config = config.get('api', {}).get(api_type, {})
    
    # Ensure required fields are present
    defaults = {
        'gpt': {
            'api_key': os.getenv('GPT_API_KEY', ''),
            'base_url': os.getenv('GPT_BASE_URL', 'https://api.gpt.ge/v1'),
            'model': os.getenv('GPT_MODEL', 'gpt-4o-mini'),
            'max_tokens': 1500,
            'temperature': 0.7,
            'max_retries': 3,
            'retry_delay': 1.0
        },
        'qwen': {
            'api_key': os.getenv('QWEN_API_KEY', ''),
            'base_url': os.getenv('QWEN_BASE_URL', 'https://api.vveai.com/v1'),
            'model': os.getenv('QWEN_MODEL', 'qwen3-14b'),
            'max_tokens': 1500,
            'temperature': 0.7,
            'max_retries': 3,
            'retry_delay': 1.0,
            'enable_thinking': False
        }
    }
    
    final_config = defaults[api_type].copy()
    final_config.update(api_config)
    return final_config

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration has required fields"""
    required_apis = ['gpt', 'qwen']
    
    for api_type in required_apis:
        api_config = get_api_config(config, api_type)
        if not api_config.get('api_key'):
            print(f"Missing API key for {api_type.upper()}")
            print(f"Set {api_type.upper()}_API_KEY environment variable")
            return False
    
    # Check data paths
    raw_data_path = config.get('data', {}).get('raw_data_path', './data/raw/original_survey_df.pkl')
    if not os.path.exists(raw_data_path):
        print(f"Raw data not found: {raw_data_path}")
        return False
    
    return True

# Example usage
if __name__ == "__main__":
    config = load_config()
    if validate_config(config):
        print("Configuration loaded successfully")
        print(f"Raw data path: {config.get('data', {}).get('raw_data_path')}")
        print(f"Output directory: {config.get('data', {}).get('output_dir')}")
    else:
        print("Configuration validation failed")
        print("Please check your .env file and data paths")
