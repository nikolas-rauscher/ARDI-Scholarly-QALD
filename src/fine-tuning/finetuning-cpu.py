
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import torch
import json
import evaluate
import nltk
import torch
import numpy as np
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

nltk.download("punkt")
metric = evaluate.load("rouge")
with open(config['FilePaths']['prompt_template']) as f:
    prompt_template = f.read()
tokenizer = None

def load_tokenizer(model_id):
    """_summary_

    Args:
        model_id (str): id of the model on HF.

    Returns:
        tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return tokenizer


def get_targets(labels_file_path):
    """_summary_

    Args:
        labels_file_path (str): path of the file which gives the name of classes.

    Returns:
        list: the list of class/label names
    """

    targets = json.load(open(labels_file_path))
    targets = [key for key in list(targets.keys())]

    return targets


def preprocess_logits_for_metrics(logits):

    if isinstance(logits, tuple):
        logits = logits[0]

    return logits.argmax(dim=-1)

# helper function to postprocess text

def extract_response(response):
    """Extract and clean the response."""
    if len(response) > 300:
        response = response.split("Answer:")[-1].strip()
        if '[INST]' in response:
            response = response.split("INST]")[-1].strip()
        if 'assistant' in response:
            response = response.split("assistant")[-1].strip()
        if len(response) > 300:
            response = response.split("\n")[-1].strip()
        if len(response) > 300:
            response = 'out of tokens'
    return response

def postprocess_text(labels, preds):
    """_summary_

    Args:
        labels (list): the list of ground truth classes
        preds (list): the list of predicted classes

    Returns:
        list, list: applied post processing to the returned answers from LLMs.

    """
    preds = [extract_response(pred)
             for pred in preds]
    labels = [extract_response(label) for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    """ Compute the metrics on evaluation. It can be extended with different metrics

    Args:
        eval_preds (list): the predictions used for evaluation

    Returns:
        dict: the dict of resulting metrics.
    """
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in the preds as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in preds]

    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    # Some simple post-processing
    grounds, preds = postprocess_text(decoded_labels, decoded_preds)

    # compute micro F1, Recall and Precision
    precision, recall, f1, s = precision_recall_fscore_support(
        grounds, preds, labels=targets, average='micro')

    decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]

    decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]

    # Compute ROUGscores
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    # Extract the median scores
    result = {key: value * 100 for key, value in result.items()}
    result["gen_len"] = np.mean(prediction_lens)

    result['f1'] = f1
    result['recall'] = recall
    result['precision'] = precision

    return {k: round(v, 4) for k, v in result.items()}


def load_model(model_id):
    """Load the model from Hugging Face

    Args:
        model_id (str): the id of model on HF

    Returns:
        model: quantized pretrained language model
    """

    compute_dtype = getattr(torch, "float16")
    # quantization configs
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # quantization_config=quant_config,
    )
    # for single GPU
    # model.config.use_cache = False
    # model.config.pretraining_tp = 1

    return model


def load_dataset_from_hub(dataset_id):
    """Load dataset from the Hugging Face hub.

    Args:
        dataset_id (str): the id of dataset on HF

    Returns:
        val_dataset, train_dataset: the validation and train datasets
    """
    # Load dataset from the hub

    train_dataset = load_dataset(dataset_id, split="train")
    test_dataset = load_dataset(dataset_id, split="test")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset


# def formatting_prompt_func(example, max_length=4096):
#     if('context' not in example):
#         example['context'] =''
#     return prompt_template.format(
#         question=example['question'], context=example['context'][:max_length-len(example['question'])-len(prompt_template)])+str(example['answer'])


def formatting_prompts_func(examples):
    """Create a list to store the formatted texts for each item in the example

    Args:
        example (list of dataset): one batch from dataset. each line might consist of prompt_template context and question
    Returns:
        prompt_texts: formated prompt_templates
    """
    global prompt_template
    prompt_texts = []
    # Iterate through each example in the batch
    for question, answer, context in zip(examples['question'], examples['answer'], examples['context']):
        # Format each example as a prompt_template-response pair
        prompt_text = prompt_template.format(
            question=question, context=context)+answer
        prompt_texts.append(prompt_text)
    # Return the list of formatted texts
    return prompt_texts


def get_cross_validation_splits(train_dataset, target_column, n_splits=5):
    """
    Get cross-validation splits for the train dataset.

    Args:
        train_dataset (Dataset): The train dataset.
        target_column (str): The name of the target column.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        List of tuples: Each tuple contains train indices and validation indices for each split.
    """
    train_df = train_dataset.to_pandas()
    X, y = train_df.drop(columns=[target_column]), train_df[target_column]

    skf = StratifiedKFold(n_splits=n_splits)
    return [(train_idx, val_idx) for train_idx, val_idx in skf.split(X, y)]


def fine_tune_model(model_id, train_dataset, val_dataset):
    """
    Fine-tune the model on the provided dataset.

    Args:
        model_id (str): The ID of the model on Hugging Face.
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.

    Returns:
        trainer: The fine-tuned trainer.
    """
    model = load_model(model_id)
    tokenizer = load_tokenizer(model_id)

    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.add_adapter(lora_config)

    training_args = TrainingArguments(
        output_dir=f"{model_id.split('/')[1]}-local",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_prompts_func,
        peft_config=lora_config,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    model = model.merge_and_unload()
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    return trainer


def load_dataset_local(path_train):
    """
    Load datasets from local JSON files.

    Args:
        path_train (str): Path to the train dataset JSON file.
        path_val (str): Path to the validation dataset JSON file.

    Returns:
        train_dataset, val_dataset: Train and validation datasets as Hugging Face Dataset objects.
    """
    train_df = pd.read_json(path_train, orient='records')
    return Dataset.from_pandas(train_df)


def perform_cross_validation(model_id, path_train, target_column, n_splits=5):
    """
    Perform cross-validation and calculate the standard error.

    Args:
        model_id (str): The ID of the model on Hugging Face.
        path_train (str): Path to the train dataset JSON file.
        path_val (str): Path to the validation dataset JSON file.
        target_column (str): The name of the target column.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        mean_accuracy, std_error: The mean accuracy and standard error.
    """
    # train_dataset, _ = load_dataset_from_hub(dataset_id=dataset_id)
    train_dataset = load_dataset_local(path_train)
    splits = get_cross_validation_splits(
        train_dataset, target_column, n_splits=n_splits)

    accuracies = []

    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"Processing fold {i + 1}/{n_splits}...")
        train_split = train_dataset.select(train_idx)
        val_split = train_dataset.select(val_idx)

        trainer = fine_tune_model(
            model_id, train_split, val_split)

        eval_results = trainer.evaluate()
        # accuracy = None
        accuracy = eval_results.get('eval_accuracy')
        if accuracy is not None:
            accuracies.append(accuracy)
            print(f"Fold {i + 1} accuracy: {accuracy}")
        else:
            print(f"Accuracy not found in fold {i + 1} results")

    mean_accuracy = np.mean(accuracies)
    std_error = np.std(accuracies) / np.sqrt(n_splits)

    print(f"Mean cross-validation accuracy: {mean_accuracy}")
    print(f"Standard error: {std_error}")

    return mean_accuracy, std_error


if __name__ == "__main__":

    model_id = "mistralai/Mistral-7B-v0.3"  # change this
    dataset_id = "Sefika/KGQA_triples"  # change this..
    path_train = config['FilePaths']['finetune_path_train']

    # target_labels = 'rel2id.json'  # file path to your target classes.
    target_column = "answer"
    trainer = perform_cross_validation(
        model_id, path_train, target_column)
