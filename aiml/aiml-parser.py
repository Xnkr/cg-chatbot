import aiml

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
        return "/Users/administrator/chatbot/cg-chatbot/aiml/salutations.xml"


"""
returns AIML match response or empty string
"""
def getResponse(query):
    responder = AIMLResponder.getAIMLResponder()
    return responder.getResponse(query)


print(getResponse("Hello"))
print(getResponse("asdff"))






# # The Kernel object is the public interface to
# # the AIML interpreter.
# k = aiml.Kernel()

# # Use the 'learn' method to load the contents
# # of an AIML file into the Kernel.
# val = k.learn(getAIMLFilePath())
# print(val)

# # Use the 'respond' method to compute the response
# # to a user's input string.  respond() returns
# # the interpreter's response, which in this case
# # we ignore.
# k.respond("hello")

# # Loop forever, reading user input from the command
# # line and printing responses.
# while True:
#     # print (k.respond(input("> ")))
#     val = k.respond(input("> "))
#     print('value returned is '+ val)
