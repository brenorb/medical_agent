import csv
import os
import pickle

from sklearn.model_selection import train_test_split

PICKLE_PATHS = ["data/rag_questions.pkl", "data/rag_answers.pkl", "data/train_questions.pkl", "data/train_answers.pkl", "data/test_questions.pkl", "data/test_answers.pkl"]
FILE_PATH = "data/intern_screening_dataset.csv"

def load_pickled_data(pickle_paths):
    """Load all pickled data into a dictionary"""
    data = dict()
    for path in pickle_paths:
        # if path does not exist, throw an error
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")
        else:
            with open(path, 'rb') as f:
                name = path.split("/")[-1].split(".")[0]
                data[name] = pickle.load(f)
                print(f"Loaded {len(data[name])} samples from {path}")
    return data

def load_csv_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    return data

def save_pickle_data(data, pickle_paths):
    for i, path in enumerate(pickle_paths):
        with open(path, 'wb') as f:
            pickle.dump(data[i], f)

try:
    data = load_pickled_data(PICKLE_PATHS)
except FileNotFoundError:
    data = load_csv_data(FILE_PATH)
    questions = [row['question'] for row in data]
    answers = [row['answer'] for row in data]
    rag_questions, remaining_questions, rag_answers, remaining_answers = train_test_split(
        questions, answers, test_size=200, random_state=42
    )
    train_questions, test_questions, train_answers, test_answers = train_test_split(
        remaining_questions, remaining_answers, test_size=0.5, random_state=42
    )
    dataset = [rag_questions, rag_answers, train_questions, train_answers, test_questions, test_answers]
    save_pickle_data(dataset, PICKLE_PATHS)
    data = {name.split(".")[0].split("/")[-1]: dataset[i] for i, name in enumerate(PICKLE_PATHS)}

if __name__ == "__main__":
    print(data['train_questions'])
