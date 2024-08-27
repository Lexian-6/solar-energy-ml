import sys
from src.logger import logging

def error_message_detail(error_message):
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"""
    Error occured in python script name {file_name}, line number {exc_tb.tb_lineno}, error message:
    {str(error_message)}
    """
    return error_message

class CustomizedException(Exception):
    def __init__(self, error_message) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message)
    
    def __str__(self) -> str:
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.error(f"An error occurs: {CustomizedException(e)}")
        raise CustomizedException(e)