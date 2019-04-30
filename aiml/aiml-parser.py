import aiml
import os

class AIMLResponder:
    """
    Singleton class to query from aiml files.
    """
    responder = None

    def __init__(self, filePath):
        self.filePath = filePath
        self.kernel = aiml.Kernel()
        self.kernel.learn(filePath)

    def getResponse(self, query):
        return self.kernel.respond(query)

    @staticmethod
    def getAIMLResponder():
        if AIMLResponder.responder is None:
            AIMLResponder.responder = AIMLResponder(
                AIMLResponder.getAIMLFilePath())
        return AIMLResponder.responder

    @staticmethod
    def getAIMLFilePath():
        return os.path.join("aiml","salutations.xml")


"""
returns AIML match response or empty string
"""
def getResponse(query):
    responder = AIMLResponder.getAIMLResponder()
    return responder.getResponse(query)


print(getResponse("Hello"))
print(getResponse("asdff"))







