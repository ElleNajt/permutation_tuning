import random
import tqdm

from datasets import load_dataset
from src.data import save_dataset, Example


def format_question(question, incorrect_answers: list[str], correct_answer: str) -> tuple[str, str]:

    all_responses = incorrect_answers + [correct_answer]
    random.shuffle(all_responses)

    question = f"""
    {question}
    (A) {all_responses[0]}
    (B) {all_responses[1]}
    (C) {all_responses[2]}
    (D) {all_responses[3]}
    """
    return question, ['A', 'B', 'C', 'D'][all_responses.index(correct_answer)]


def format_example(example):
    question, answer = format_question(
        example['Question'], 
        [example['Incorrect Answer 1'], example['Incorrect Answer 2'], example['Incorrect Answer 3']], 
        example['Correct Answer']
    )
    return Example(
        question = question,
        answer = answer,
        cot = example['Explanation'],
    )

for dataset in ['gpqa_main', 'gpqa_diamond']:
    raw_data = load_dataset("Idavidrein/gpqa", dataset)['train']
    print('Data Loaded', len(raw_data))


    data = []
    for ex in tqdm.tqdm(raw_data):
        data.append(format_example(ex))


    save_dataset(data, f'results/datasets/{dataset}_train.json')