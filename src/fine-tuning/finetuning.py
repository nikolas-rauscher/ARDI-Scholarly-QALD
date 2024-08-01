from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from sklearn.metrics import precision_recall_fscore_support
import torch
import json
import evaluate
import nltk
import numpy as np

nltk.download("punkt")

metric = evaluate.load("rouge")

def load_tokenizer(model_id):
    """
    Load and configure the tokenizer.

    Args:
        model_id (str): Identifier of the model on Hugging Face.

    Returns:
        AutoTokenizer: The tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return tokenizer

def get_targets(labels_file_path):
    """
    Retrieve the list of class/label names from a JSON file.

    Args:
        labels_file_path (str): Path to the file containing class/label names.

    Returns:
        list: A list of class/label names.
    """
    with open(labels_file_path) as f:
        targets = json.load(f)
    return list(targets.keys())

def preprocess_logits_for_metrics(logits):
    """
    Process the logits to obtain predicted class indices.

    Args:
        logits: The logits output from the model.

    Returns:
        torch.Tensor: The class indices with the highest logit values.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def postprocess_text(labels, preds):
    """
    Post-process the text output from the model predictions.

    Args:
        labels (list): List of ground truth classes.
        preds (list): List of predicted classes.

    Returns:
        tuple: Processed labels and predictions.
    """
    preds = [pred.replace('\n', '').split('Answer:')[-1].strip() for pred in preds]
    labels = [label.replace('\n', '').split('Answer:')[-1].strip() for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    """
    Compute evaluation metrics based on predictions and ground truths.

    Args:
        eval_preds (list): The list of predictions and ground truth labels.

    Returns:
        dict: A dictionary of computed metric scores.
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    grounds, preds = postprocess_text(decoded_labels, decoded_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(grounds, preds, labels=targets, average='micro')

    decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value * 100 for key, value in result.items()}
    result["gen_len"] = np.mean(prediction_lens)
    result['f1'] = f1
    result['recall'] = recall
    result['precision'] = precision

    return {k: round(v, 4) for k, v in result.items()}

def load_model(model_id):
    """
    Load a quantized pretrained language model.

    Args:
        model_id: Identifier of the model on Hugging Face.

    Returns:
        AutoModelForCausalLM: The loaded model instance.
    """
    compute_dtype = torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map={"": 0}
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model

def load_dataset_from_hub(dataset_id):
    """
    Load the dataset from Hugging Face Hub.

    Args:
        dataset_id (str): Identifier of the dataset on Hugging Face.

    Returns:
        tuple: Training and validation datasets.
    """
    train_dataset = load_dataset(dataset_id, split="train")
    val_dataset = load_dataset(dataset_id, split="validation")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    return train_dataset, val_dataset

def formatting_prompts_func(example):
    """
    Format the prompts for each example in the dataset.

    Args:
        example (dict): A batch of examples from the dataset.

    Returns:
        list: List of formatted prompts.
    """
    formatted_texts = []
    for text, raw_label in zip(example['prompt'], example['relation']):
        formatted_text = f"[INST] {text} [\INST] Answer:{raw_label}"
        formatted_texts.append(formatted_text)
    return formatted_texts

def fine_tuning(model_id, dataset_id, target_column):
    """
    Fine-tune the language model on the specified dataset.

    Args:
        model_id (str): Identifier of the model on Hugging Face.
        dataset_id (str): Identifier of the dataset on Hugging Face.
        target_labels (str): Path to the file containing target class names.

    Returns:
        SFTTrainer: The fine-tuned model trainer instance.
    """
    print(f"Fine-tuning model: {model_id} on dataset: {dataset_id}")

    train_dataset, val_dataset = load_dataset_from_hub(dataset_id)
    global targets
    targets = get_targets(target_labels)
    model = load_model(model_id)

    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    repository_id = f"{model_id.split('/')[1]}-{dataset_id}"

    model.add_adapter(lora_config)
    global tokenizer
    tokenizer = load_tokenizer(model_id)

    training_args = TrainingArguments(
        output_dir=repository_id,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
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
        max_seq_length=None,
        packing=False,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    model = model.merge_and_unload()
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    return trainer

if __name__ == "__main__":
    model_id = "google/flan-t5-large"
    dataset_id = "wepolyu/KGQA"
    trainer = fine_tuning(model_id, dataset_id, target_column="answer")
