import functools
import logging
import os

def get_logger(log_path: str) -> logging.Logger:
    """ Creates a Log File and returns Logger object """
    
    # Load configuration
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if os.path.exists(log_path):
        message = f"Log file already exists at {log_path}. Appending to existing log file."
    else:
        message = f"Creating new log file at {log_path}."
    
    logging.basicConfig(level=logging.INFO, format=log_fmt, filename=log_path, force=True)
    
    logger = logging.getLogger(__name__)
    
    logger.info(message)

    # Return logger object
    return logger


def log(_func=None, *, my_logger: logging.Logger = None):
    def decorator_log(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if my_logger is None:
                logger = get_logger(log_path="logs/main.log")
            else:
                logger = my_logger
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            logger.info(f"function {func.__name__} called with args {signature}")
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                logger.exception(f"Exception raised in {func.__name__}. exception: {str(e)}")
                raise e
        return wrapper

    if _func is None:
        return decorator_log
    else:
        return decorator_log(_func)