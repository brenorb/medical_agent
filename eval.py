import dspy
from dspy.evaluate import Evaluate

from agent import Agent
from preprocess import data

train, test = [], []
for q, a in zip(data['train_questions'], data['train_answers']):
    train.append(dspy.Example(medical_question=q, answer=a).with_inputs("medical_question"))

for q, a in zip(data['test_questions'], data['test_answers']):
    test.append(dspy.Example(medical_question=q, answer=a).with_inputs("medical_question"))


# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()

def metric(gold, pred, trace=None):
    question, answer = gold.medical_question, gold.answer

    # correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"
    correct = f"The text answer `{question}`. Does the assessed text contain this answer?"

    correct =  dspy.Predict(Assess)(assessed_text=pred, assessment_question=correct)

    return correct.assessment_answer


agent = Agent()

evaluate = Evaluate(metric=metric, devset=train, num_threads=1, display_progress=True, display_table=5)
evaluate(agent)


# optimizer = dspy.MIPROv2(metric=metric)
# optimized_agent = optimizer.compile(agent, trainset=train)
# optimized_agent.save(path="/tmp/model.json")




if __name__ == "__main__":
    # lm = dspy.LM(model="openai/gpt-4o-mini")
    # dspy.configure(lm=lm)
    # ex = dspy.Example(medical_question="What is (are) High Blood Pressure ?", answer="Blood pressure is the force of blood pushing against the walls of the blood vessels as the heart pumps blood. If your blood pressure rises and stays high over time, its called high blood pressure. High blood pressure is dangerous because it makes the heart work too hard, and the high force of the blood flow can harm arteries and organs such as the heart, kidneys, brain, and eyes.")
    # pred = "High blood pressure is a common disease in which blood flows through blood vessels (arteries) at higher than normal pressures. There are two main types of high blood pressure: primary and secondary high blood pressure. Primary, or essential, high blood pressure is the most common type of high blood pressure. This type of high blood pressure tends to develop over years as a person ages. Secondary high blood pressure is caused by another medical condition or use of certain medicines. This type usually resolves after the cause is treated or removed."
    # from agent import Agent
    # agent = Agent()
    # hist = dspy.History(messages=[])
    # response = agent.forward(question=train[0]['medical_question'], history=hist)
    # print(response, train[0])
    # print(metric(train[0], response))
    # print(metric(ex, pred))
    # print(dspy.inspect_history())
    pass
