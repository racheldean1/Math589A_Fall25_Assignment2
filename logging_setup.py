# File: logging_setup.py
# Insert this code at the top of your implementation file.
# Alternatively, import this file as a module (even better).
# (most likely, general_linear_solver.py) to enable logging.
# REMEMBER: Don't print, log.

import numpy as np

# level: Sets the minimum severity level for messages to be processed
# (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). Messages below this
# level will be ignored.
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s',
#                     handlers=[logging.FileHandler("app.log"),
#                               logging.StreamHandler()])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logger's level
