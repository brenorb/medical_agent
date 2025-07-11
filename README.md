# Medical Question Answering System

A medical question answering system built with DSPy that can answer medical questions using AI. The system includes training, evaluation, and optimization components.

## Overview

This project implements a medical QA agent that:
- Answers medical questions using OpenAI's GPT-4o-mini
- Uses DSPy framework for structured prompting and optimization
- Includes training data preprocessing and evaluation metrics
- Supports conversation history for context-aware responses


## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key

### Installation

1. Clone the repository and navigate to the project directory
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Set up environment variables:
   
## Usage

### Interactive Chat Mode

Run the agent in interactive mode:

```bash
python agent.py
```
Type 'q' to quit.


### Training and Evaluation

The model as is performs with ~92% accuracy on the test set. The evaluation metric is a simple LLM-as-judge asking if the question is answered correctly.
I didn't have time to optimize the model (although the code is there) and think more about the evaluation system.


### Jupyter Notebook Examples

See `examples.ipynb` for example interactions.

## Data

The system uses a medical question-answer dataset that includes:
- **RAG corpus**: ~16K question-answer pairs for retrieval
- **Training set**: 100 question-answer pairs for training
- **Test set**: 100 question-answer pairs for evaluation

Data is automatically preprocessed and split when first run.

The system would use the RAG corpus as context to answer the questions but I didn't have time to implement it. I am using only 100 question-answer pairs for training and evaluation because it can get too slow and expensive to train (optmize the prompts) and evaluate on more than that.

## Configuration

The system uses DSPy configuration with OpenAI's GPT-4o-mini model. You can modify the model in `agent.py`:

```python
lm = dspy.LM(model="openai/gpt-4o-mini")  # Change model here
dspy.configure(lm=lm)
```

## Dependencies

Main dependencies:
- `dspy-ai`: DSPy framework for LM applications
- `scikit-learn`: For data splitting
- `jupyter`: For notebook examples
- `python-dotenv`: Environment variable management


## License

This project is licensed under the MIT License.

## Notes

- Ensure you have a valid OpenAI API key configured
- The system requires internet connection for OpenAI API calls
- CSV data file `intern_screening_dataset.csv` should be inside `data` folder