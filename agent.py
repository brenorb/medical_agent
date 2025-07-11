from typing import Optional

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM(model="openai/gpt-4o-mini")
dspy.configure(lm=lm)

class QASignature(dspy.Signature):
    """
    Answer medical questions based on the context.
    """
    medical_question: str = dspy.InputField(description="The medical question to answer")
    history: Optional[dspy.History] = dspy.InputField(description="The history of the conversation")
    context: Optional[str] = dspy.InputField(description="The context to use to answer the question")
    # Output fields
    answer: str = dspy.OutputField(description="The answer to the question")

class Agent(dspy.Module):
    def __init__(self):
        self.qa = dspy.ChainOfThought(QASignature)

    def forward(self, medical_question: str, history: Optional[dspy.History] = None) -> str:
        # search for context
        context = None # TODO: search for context
        if history is None:
            history = dspy.History(messages=[])
        
        result = self.qa(medical_question=medical_question, history=history, context=context)
        return result.answer



if __name__ == "__main__":
    COLOR_USER = "\033[94m"
    COLOR_ASSISTANT = "\033[92m"
    COLOR_RESET = "\033[0m"

    agent = Agent()
    history = dspy.History(messages=[])
    
    while True:
        user_message = input(f"{COLOR_USER}User: {COLOR_RESET}")
        if user_message.lower() == "q":
            print("Bye!")
            break

        response = agent.forward(user_message, history)
        history.messages.append({"question": user_message, "answer": response})
        print(f"{COLOR_ASSISTANT}{response}{COLOR_RESET}")
