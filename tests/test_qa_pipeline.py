# coding: utf-8
from models.qa_pipeline import qa_with_merged_data, question_answering
import random
from datasets import load_dataset


def ask_question():
    """
    Prompts the user to either request a random question or enter a natural language question and author's DBLP URI.

    Returns:
        tuple: A tuple containing the natural question and the author's DBLP URI. If a random question is requested, both values are None.
    """
    user_input = input("Would you like a random question? (y/n): ").lower()
    if user_input == 'y':
        test_random_question()
    elif user_input == 'n':
        natural_question = input("Please enter a natural language question: ")
        print("You entered the question:", natural_question)
        author_dblp_uri = input(
            "Please enter the author's DBLP URI (format: https://dblp.org/pid/w/TDWilson): ")
        print("Author's DBLP URI:", author_dblp_uri)
        test_natural_question(natural_question, author_dblp_uri)
    else:
        print("Invalid input. Please enter y or n.")
        ask_question()


def test_random_question():
    dataset = load_dataset('wepolyu/QALD-2024')
    random_index = random.randint(0, len(dataset['train']) - 1)
    item = dataset['train'][random_index]
    question = item["question"]
    # Removing leading and trailing brackets
    authors = item["author_dblp_uri"][1:-1]
    answer = question_answering(question, [authors])
    print(f'Question: {question}\nRandom Question Answer: {answer}')


def test_natural_question(natural_question, author_dblp_uri):
    answer = question_answering(natural_question, author_dblp_uri)
    print("\nAnswer: " + answer)


if __name__ == "__main__":
    ask_question()
