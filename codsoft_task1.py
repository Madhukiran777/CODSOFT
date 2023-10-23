# Importing modules
import re

# Define a dictionary of patterns and responses
patterns = {
    r"hi|hello|hey": "Hello, how are you?",
    r"how are you": "I'm doing well, thank you.",
    r"name": "My name is leo, created by Madhu kiran.",
    r"what can you do": "I can help you with some questions and tasks.",
    r"weather": "It's too hot in your area.",
    r"favorite color": "I do not have a specific favorite color, I like all colors",
    r"bye|goodbye|see you": "Goodbye, have a nice day."
}

# function to match the user input with a pattern
def match_pattern(user_input):
    for pattern, response in patterns.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return response
    return "Sorry, I don't understand."

# function to run the chatbot
def run_chatbot():
    print("Welcome to the chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        chatbot_response = match_pattern(user_input)
        # Print the chatbot response
        print("Chatbot: " + chatbot_response)

# Run the chatbot
run_chatbot()
