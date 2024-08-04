
import configparser
from models.zero_shot_prompting_pipeline import run_model


config = configparser.ConfigParser()
config.read('config.ini')

if __name__ == '__main__':
    api = config['Flags']['use_api'] == 'False'
    api_type = config['Flags']['api_type']

    model_templates = {
        config['Model']['model_id_1']: config['Templates']['model_1_prompt_templates'].split(', '),
    }

    for model_id, templates in model_templates.items():
        model_name = model_id.split('/')[-1]
        run_model(model_id, model_name, templates, api=api, api_type=api_type)
