import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'data', 'utils', 'contracts']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        with open('.env.example', 'r') as example_file:
            example_content = example_file.read()
        with open('.env', 'w') as env_file:
            env_file.write(example_content)
        print("Created .env file from .env.example")

if __name__ == "__main__":
    print("Setting up the Credit Risk Verification System...")
    install_requirements()
    create_directories()
    create_env_file()
    print("Setup completed successfully!")
    print("Next steps:")
    print("1. python create_sample_data.py (if you don't have data)")
    print("2. python train_model.py")
    print("3. streamlit run app.py")
