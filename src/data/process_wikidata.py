import ast
import json
path = "data/external/wiki_data.txt"
with open(path, "r") as fout:
    data = fout.readlines()


def format_man(input_str:str):
    json_str = input_str.replace("'", "")
    json_str = json_str.replace('"', "")
    json_str = json_str.replace("\\n", "")
    json_str = json_str.replace('\\', '')
    json_str = json_str.replace("institute_wikipedia_text:", '"institute_wikipedia_text":"')
    json_str = json_str.replace("'author_wikipedia_text'", '"author_wikipedia_text":"')
    json_str= json_str+ '"}]'
    return json_str


data_dict = []
for i in range(len(data)):
    try:
        output = ast.literal_eval(data[i])
        output = ast.literal_eval(output) 
        data_dict.append(output)
    except:
        #print("fail in:", i)
        output = format_man(output)
        output = json.loads(output)
        data_dict.append(output)


    with open("data/external/wiki_data_processed.txt", 'w') as file:
        json.dump(data_dict, file, indent=4, ensure_ascii=False)
