import logging
import os
from datetime import datetime

log_dir = "logs"
os.makedirs(log_dir, exist_ok= True)

formatted_datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
file_name = os.path.join(log_dir, f"app_{formatted_datetime}.log")

logging.basicConfig(
    filename=file_name,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
