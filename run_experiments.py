#!/usr/bin/env python3
"""
Survey Generation Experiment Runner

This script provides a unified interface to run all survey generation experiments.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.utils.config_loader import load_config, validate_config

def run_experiment(experiment_name, num_surveys=2):
    """Run a specific experiment"""
    
    experiments = {
        "gpt_main": {
            "script": "experiments/main/survey_generation_gpt.py",
            "description": "Main survey generation using GPT-4o-mini"
        },
        "qwen_main": {
            "script": "experiments/main/survey_generation_qwen.py", 
            "description": "Main survey generation using Qwen3-14B"
        },
        "qwen_no_rag": {
            "script": "experiments/qwen_ablations/qwen_ablation_no_rag.py",
            "description": "Qwen without RAG ablation study"
        },
        "qwen_single_agent": {
            "script": "experiments/qwen_ablations/qwen_ablation_single_agent.py",
            "description": "Qwen single agent ablation study"
        },
        "evaluate_qwen": {
            "script": "experiments/evaluations/evaluate_qwen_outputs.py",
            "description": "Evaluate Qwen generated outputs"
        }
    }
    
    if experiment_name not in experiments:
        print(f"Error: Unknown experiment '{experiment_name}'")
        print("Available experiments:")
        for name, info in experiments.items():
            print(f"  {name}: {info['description']}")
        return False
    
    script_path = experiments[experiment_name]["script"]
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found: {script_path}")
        return False
    
    print(f"Running {experiment_name}: {experiments[experiment_name]['description']}")
    print("-" * 50)
    
    # Debug: Print API configuration
    print("🔍 API Configuration Check:")
    print(f"  GPT_API_KEY: {os.getenv('GPT_API_KEY', 'NOT SET')}")
    print(f"  GPT_BASE_URL: {os.getenv('GPT_BASE_URL', 'NOT SET')}")
    print(f"  GPT_MODEL: {os.getenv('GPT_MODEL', 'NOT SET')}")
    print(f"  QWEN_API_KEY: {os.getenv('QWEN_API_KEY', 'NOT SET')}")
    print(f"  QWEN_BASE_URL: {os.getenv('QWEN_BASE_URL', 'NOT SET')}")
    print(f"  QWEN_MODEL: {os.getenv('QWEN_MODEL', 'NOT SET')}")
    print("-" * 50)
    
    # Change to project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Set environment variable for number of surveys
        env = os.environ.copy()
        if num_surveys:
            env['NUM_SURVEYS'] = str(num_surveys)
            
        result = subprocess.run([
            sys.executable, script_path
        ], env=env, check=True)
        
        print(f"✅ {experiment_name} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {experiment_name} failed with exit code {e.returncode}")
        return False

def setup_directories():
    """Create necessary directories"""
    dirs = [
        "data/raw",
        "data/processed", 
        "outputs",
        "logs"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✅ Directories created")

def main():
    parser = argparse.ArgumentParser(description="Run survey generation experiments")
    parser.add_argument("experiment", 
                       choices=["gpt_main", "qwen_main", "qwen_no_rag", 
                               "qwen_single_agent", "evaluate_qwen", "all"],
                       nargs='?',  # Make experiment optional
                       help="Experiment to run")
    parser.add_argument("--num-surveys", type=int, default=2,
                       help="Number of surveys to generate")
    parser.add_argument("--setup", action="store_true",
                       help="Setup directories and check requirements")
    parser.add_argument("--check-config", action="store_true",
                       help="Check configuration without running experiments")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Handle configuration check
    if args.check_config:
        if validate_config(config):
            print("✅ Configuration is valid!")
            return
        else:
            print("❌ Configuration validation failed")
            print("   Please check your .env file and data paths")
            sys.exit(1)
    
    # Handle setup
    if args.setup:
        setup_directories()
        print("Setup complete!")
        return
    
    # Check if experiment is provided
    if not args.experiment:
        parser.print_help()
        print("\n💡 Tip: Use --check-config to validate your setup before running experiments")
        sys.exit(1)
    
    # Validate configuration before running experiments
    if not validate_config(config):
        print("❌ Configuration validation failed")
        print("   Please check your .env file and data paths")
        sys.exit(1)
    
    if args.experiment == "all":
        experiments = ["gpt_main", "qwen_main", "qwen_no_rag", "qwen_single_agent"]
        for exp in experiments:
            print(f"\n{'='*60}")
            run_experiment(exp, args.num_surveys)
    else:
        run_experiment(args.experiment, args.num_surveys)

if __name__ == "__main__":
    main()