import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format = '[%(asctime)s - %(message)s]')

def create_project_template():
    list_files = [
        "data/input",
        "data/processed",
        "notebooks/EDA.ipynb",
        "src/data_components/__init__.py",
        "src/data_components/data_injection_scripts.py",
        "src/data_components/data_preprocessing_scripts.py",
        "src/data_components/model_evaluation_scripts.py",
        "src/logger.py",
        "src/exception.py",
        "src/utils.py",
        "src/__init__.py",
        "setup.py",
        "requirements.txt",
        "app.py",
        "main.py",
    ]

    logging.info(f"Creating project template")
    
    for file_path in list_files:
        file_path = Path(file_path)
        directory, filename = os.path.split(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(file_path, "w") as f:
            if filename.endswith(".py"):
                f.write("# This is an empty Python file.")
            elif filename.endswith(".ipynb"):
                f.write("# This is an empty Jupyter Notebook.")
                
    logging.info(f"Project template created successfully")
                
create_project_template()
