import configparser

from models.predictions_pipeline import run_model, run_model4settings


config = configparser.ConfigParser()
config.read('config.ini')


def test_predictions1setting():
    model_templates = {
        config['Model']['model_id_1']: config['Templates']['model_1_prompt_templates'].split(', ')
    }

    for model_id, templates in model_templates.items():
        model_name = model_id.split('/')[-1]
        run_model(model_id, model_name, templates)

def test_predictions4settings():
    model_templates = {
        config['Model']['model_id_1']: config['Templates']['model_1_prompt_templates'].split(', ')
    }

    for model_id, templates in model_templates.items():
        model_name = model_id.split('/')[-1]
        run_model4settings(model_id, model_name, templates)

if __name__ == '__main__':
    test_predictions1setting()
    test_predictions4settings()