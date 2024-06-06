def formatting_prompts_func(prompt, example):
    """Create a list to store the formatted texts for each item in the example

    Args:
        example (list of dataset): one batch from dataset. each line might consist of prompt context and question
    Returns:
        formatted_texts: formated prompts
    """

    formatted_texts = []

    # Iterate through each example in the batch
    for context, question in zip(example['context'], example['question']):
        # Format each example as a prompt-response pair
        formatted_text = prompt.replace('[question]', question)
        formatted_text = prompt.replace('[context]', "Context: "+context)
        formatted_texts.append(formatted_text)
    # Return the list of formatted texts
    return formatted_texts
