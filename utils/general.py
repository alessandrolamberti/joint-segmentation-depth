import logging
import time
from functools import wraps

import yaml

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logging.info(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result
    return timeit_wrapper

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
