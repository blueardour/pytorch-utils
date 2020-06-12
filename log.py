import shutil
import os
import logging.config
import json
import pandas as pd

def setup_logging(log_file='info.txt', resume=False, dummy=False, stdout=True):
    """
    Setup logging configuration
    """
    if dummy:
        logging.getLogger('dummy')
    else:
        if os.path.isfile(log_file) and resume:
            file_mode = 'a'
        else:
            file_mode = 'w'

        logging.shutdown() # shutdown all logging before
        root_logger = logging.getLogger()
        if root_logger.handlers:
            root_logger.handlers[0].close()
            root_logger.removeHandler(root_logger.handlers[0])
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            filename=log_file,
                            filemode=file_mode)
        if stdout:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)


