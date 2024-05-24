
import torch
import json
import evaluate
import nltk, torch
import numpy as np
nltk.download("punkt")

from sklearn.metrics import precision_recall_fscore_support
from peft import LoraConfig

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

from trl import SFTTrainer
from datasets import load_dataset

metric = evaluate.load("rouge")


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
def postprocess_text(labels, preds):
    """_summary_

    Args:
        labels (list): the list of ground truth classes
        preds (list): the list of predicted classes

    Returns:
        list, list: applied post processing to the returned answers from LLMs.

    """
    preds = [pred.replace('\n','').split('Answer:')[-1].strip() for pred in preds]
    labels = [label.replace('\n','').split('Answer:')[-1].strip() for label in labels]

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
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    # Some simple post-processing
    grounds, preds = postprocess_text(decoded_labels,decoded_preds)

    # compute micro F1, Recall and Precision
    precision, recall, f1, s = precision_recall_fscore_support(grounds, preds, labels=targets, average='micro')
    
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
    #quantization configs
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
    #for single GPU
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
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
    val_dataset = load_dataset(dataset_id, split="validation")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    
    return train_dataset, val_dataset


def formatting_prompts_func(example):
    """Create a list to store the formatted texts for each item in the example

    Args:
        example (list of dataset): one batch from dataset. each line might consist of prompt context and target_label
    Returns:
        formatted_texts: formated prompts
    """
     
    formatted_texts = []

    # Iterate through each example in the batch
    for text, raw_label in zip(example['prompt'], example['relation']):
        # Format each example as a prompt-response pair
        formatted_text = f"[INST] {text} [\INST] Answer:{raw_label}"
        formatted_texts.append(formatted_text)
    # Return the list of formatted texts
    return formatted_texts

def fine_tuning(model_id, dataset_id, target_labels):
    """_summary_

    Args:
        model_id (str): the id of model on Hugging Face
        dataset_id (str): the id of dataset on HF
        target_labels (list): the names of classes in the dataset

    Returns:
        trainer: SFT trainer. The trainer can be used for future predictions later.
    """

    print("Fine tuning model: ", model_id, " on dataset: ", dataset_id)

    train_dataset, val_dataset = load_dataset_from_hub(dataset_id)
    global targets 
    targets = get_targets(target_labels)
    # quantized pretrained model
    model = load_model(model_id)

    # apply LoRA configuration for CAUSAL LM, decode only models, such as Llama2-7B and Mistral-7B
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


    #declare training arguments
    #please change it for more than one epoch. such as add val_loss for evaluation on epoch..
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


    # declare trainer
    trainer = SFTTrainer( #based on RLHF
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            formatting_func=formatting_prompts_func,
            peft_config=lora_config,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,        
            max_seq_length= None,
            packing=False,
        )

    trainer.train()
    
    #save trainer
    trainer.save_model(training_args.output_dir)
    # merge adapter and pretrained weights
    model = model.merge_and_unload()
    #save fine-tuned model
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    return trainer

if __name__ == "__main__":
    
    model_id = "mistralai/Mistral-7B-Instruct-v0.2" #change this
    dataset_id = "Sefika/tacrev_prompt" #change this..

    target_labels = 'rel2id.json' #file path to your target classes.
    trainer = fine_tuning(model_id, dataset_id, target_labels )


