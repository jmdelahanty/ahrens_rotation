class ConfigError(Exception):
    """
    Exception for when there's an error with Config Files.
    """
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "TemplateError: " + "{0}".format(self.message)
        else:
            return "TEMPLATE ERROR"

class VideoReadError(Exception):
    """Custom exception for video read errors."""
    pass

class VideoProcessingWarning(Warning):
    """Custom warning for video processing issues."""
    pass