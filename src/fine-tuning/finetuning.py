from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
import pandas as pd
import torch
import json
import evaluate
import nltk
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

nltk.download("punkt")
metric = evaluate.load("rouge")
with open(config['FilePaths']['prompt_template']) as f:
    prompt_template = f.read()

def get_cross_validation_splits(train_dataset, target_column, n_splits=5):
    """
    Get cross-validation splits for the train dataset.

    Args:
        train_dataset (Dataset): The train dataset.
        target_column (str): The name of the target column.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        list: A list of tuples, each containing train indices and validation indices for each split.
    """
    train_df = train_dataset.to_pandas()
    X, y = train_df.drop(columns=[target_column]), train_df[target_column]

    skf = StratifiedKFold(n_splits=n_splits)
    return [(train_idx, val_idx) for train_idx, val_idx in skf.split(X, y)]

# ==================================================================================
# ==============Functions basically never called externally=========================
# ==================================================================================

def load_tokenizer(model_id):
    """
    Load the tokenizer from Hugging Face.

    Args:
        model_id (str): The ID of the model on Hugging Face.

    Returns:
        tokenizer: The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return tokenizer


def get_targets(labels_file_path):
    """
    Get the list of target class labels from a file.

    Args:
        labels_file_path (str): Path to the file containing class labels.

    Returns:
        list: The list of class/label names.
    """
    targets = json.load(open(labels_file_path))
    return [key for key in targets.keys()]


def preprocess_logits_for_metrics(logits):
    """
    Preprocess the logits for metric computation.

    Args:
        logits (torch.Tensor): The model's output logits.

    Returns:
        torch.Tensor: The indices of the maximum logits.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def extract_response(response):
    """
    Extract and clean the response from the model output.

    Args:
        response (str): The raw response from the model.

    Returns:
        str: The cleaned response.
    """
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
    """
    Post-process the labels and predictions.

    Args:
        labels (list): The list of ground truth labels.
        preds (list): The list of predicted labels.

    Returns:
        tuple: The post-processed labels and predictions.
    """
    preds = [extract_response(pred) for pred in preds]
    labels = [extract_response(label) for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    """
    Compute the evaluation metrics for the model predictions.

    Args:
        eval_preds (list): The predictions used for evaluation.

    Returns:
        dict: The dictionary containing the calculated metrics.
    """
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in preds]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    grounds, preds = postprocess_text(decoded_labels, decoded_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        grounds, preds, average='micro'
    )

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}
    result["gen_len"] = np.mean(prediction_lens)
    result['f1'] = f1
    result['recall'] = recall
    result['precision'] = precision

    return {k: round(v, 4) for k, v in result.items()}


def load_model(model_id):
    """
    Load the model from Hugging Face.

    Args:
        model_id (str): The ID of the model on Hugging Face.

    Returns:
        model: The loaded quantized language model.
    """
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model


def load_dataset_from_hub(dataset_id):
    """
    Load dataset from the Hugging Face hub.

    Args:
        dataset_id (str): The ID of the dataset on Hugging Face.

    Returns:
        tuple: The validation and training datasets.
    """
    train_dataset = load_dataset(dataset_id, split="train")
    test_dataset = load_dataset(dataset_id, split="test")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset


def formatting_prompts_func(examples):
    """
    Format the prompts for training.

    Args:
        examples (dict): The batch of data containing questions, answers, and context.

    Returns:
        list: The formatted prompt texts.
    """
    global prompt_template
    prompt_texts = []
    for question, answer, context in zip(examples['question'], examples['answer'], examples['context']):
        prompt_text = prompt_template.format(
            question=question, context=context) + answer
        prompt_texts.append(prompt_text)
    return prompt_texts


def fine_tune_model(model_id, train_dataset, val_dataset):
    """
    Fine-tune the model on the provided dataset.

    Args:
        model_id (str): The ID of the model on Hugging Face.
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.

    Returns:
        SFTTrainer: The fine-tuned trainer.
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

    Returns:
        Dataset: The train dataset as a Hugging Face Dataset object.
    """
    train_df = pd.read_json(path_train, orient='records')
    return Dataset.from_pandas(train_df)


def perform_cross_validation(model_id, path_train, target_column, n_splits=5):
    """
    Perform cross-validation and calculate the standard error.

    Args:
        model_id (str): The ID of the model on Hugging Face.
        path_train (str): Path to the train dataset JSON file.
        target_column (str): The name of the target column.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        tuple: The mean accuracy and standard error.
    """
    train_dataset = load_dataset_local(path_train)
    splits = get_cross_validation_splits(
        train_dataset, target_column, n_splits=n_splits)

    accuracies = []

    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"Processing fold {i + 1}/{n_splits}...")
        train_split = train_dataset.select(train_idx)
        val_split = train_dataset.select(val_idx)

        trainer = fine_tune_model(model_id, train_split, val_split)
        eval_results = trainer.evaluate()
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
    model_id = "google/flan-t5-large"
    dataset_id = "Sefika/KGQA_triples"
    # path_train = config['FilePaths']['finetune_path_train']

    target_column = "answer"
    # trainer = perform_cross_validation(model_id, path_train, target_column)
    train_dataset = load_dataset(
        "wepolyu/old_60001_prompt", split="train")

    KGQA = get_cross_validation_splits(
        train_dataset, target_column, n_splits=5)

    for idx, (train_idx, val_idx) in enumerate(KGQA):
        df_subset = pd.DataFrame(train_dataset[train_idx])
        df_subset2 = pd.DataFrame(train_dataset[val_idx])
        # Save the subset DataFrame to a JSON file
        df_subset.to_json(
            f"data/processed/splits/train_{idx}.json", orient='records', lines=False)
        df_subset2.to_json(
            f"data/processed/splits/test_{idx}.json", orient='records', lines=False)
