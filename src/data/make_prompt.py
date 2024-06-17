def formatting_prompts_func(prompt_template, example):
    """Create a list to store the formatted texts for each item in the example

    Args:
        example (list of dataset): one batch from dataset. each line might consist of prompt_template context and question
    Returns:
        prompt_texts: formated prompt_templates
    """

    prompt_texts = []
    # Iterate through each example in the batch
    for context, question, raw_label in zip(example['context'], example['question'], example['relation']):
        # Format each example as a prompt_template-response pair
        prompt_text = prompt_template.format(
            question=question, context=context)
        prompt_texts.append(prompt_text+raw_label)
    # Return the list of formatted texts
    return prompt_texts
