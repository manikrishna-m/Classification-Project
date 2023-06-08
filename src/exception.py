class CustomException(Exception):
    def __init__(self, error_message, sys_module):
        _, _, self.traceback = sys_module.exc_info()
        self.error_message = error_message
        self.sys_module = sys_module
        self.error_type = type(sys_module.exc_info()[1]).__name__
        super().__init__(self.get_full_message(sys_module))

    def get_full_message(self):
        exc_type, exc_obj, exc_tb = self.sys_module.exc_info()
        filename = exc_tb.tb_frame.f_code.co_filename
        line = exc_tb.tb_lineno
        return f"Error ({self.error_type}) occurred in file '{filename}', line {line}: {self.error_message}"

    def __str__(self):
        return self.get_full_message()
