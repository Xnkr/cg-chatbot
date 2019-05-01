import os

import aiml

try:
    from static.constants import *
except:
    print('Executing without static')


class AIMLResponder:
    """
    Singleton class to query from aiml_parser files.
    """
    responder = None

    def __init__(self, filePath):
        self.filePath = filePath
        self.kernel = aiml.Kernel()
        self.kernel.verbose(isVerbose=DEBUG)
        self.kernel.learn(filePath)

    def get_response(self, query):
        return self.kernel.respond(query)

    @staticmethod
    def get_aiml_responder():
        if AIMLResponder.responder is None:
            AIMLResponder.responder = AIMLResponder(
                AIMLResponder.get_aiml_file_path())
        return AIMLResponder.responder

    @staticmethod
    def get_aiml_file_path():
        return os.path.join(AIML_DIR, AIML_FILE)


"""
returns AIML match response or empty string
"""


def get_response(query):
    responder = AIMLResponder.get_aiml_responder()
    return responder.get_response(query)


def get_kernel():
    kernel = aiml.Kernel()
    kernel.verbose(True)
    kernel.learn(os.path.join(AIML_DIR, AIML_FILE))
    return kernel


if __name__ == '__main__':
    kernel = get_kernel()
    print(kernel.respond("hello"))
    print(kernel.respond("asdff"))
