import os
from datasets import load_dataset  # huggingface datasets
import numpy as np
import huggingface_hub

input_path = "/prodcpfs/user/fengmingquan/dataset/raw/synthetic-humans-1m"


num_proc = 40
num_proc_load_dataset = 40
key_file = "/cpfs/user/fengmingquan/nanoGPT/hf_key.txt"

def gen_qa_data(output_path):

    dataset = load_dataset(input_path, num_proc=num_proc_load_dataset)

    split_dataset = dataset["train"].train_test_split(test_size=0.01, shuffle=False)
    
    print(split_dataset)
    
    #print an example
    #print(split_dataset['train'][0])
    #raise Exception("stop")
    # we now want to tokenize the dataset. first define the encoding function
    def process(example):

        out = {}
        if output_path.endswith("gender"):
            out["question"] = f"What is the gender of {example['id']}"
            choices = ["Male", "Female", "Other", "Unknown"]
            out["answer_text"] = example['gender']
        else:
            field_dict = {
                "age": "age",
                "location": "location",
                "occupation": "occupation_category",
                "wage": "annual_wage",
            }
            field_name = output_path.split('-')[-1]
            field_key = field_dict.get(field_name)
            out["question"] = f"What is the {field_name} of {example['id']}"
            wrong_answers = []
            num_choices = 4
            while len(wrong_answers) < num_choices - 1:
                wrong_example = split_dataset['train'][np.random.randint(0, len(split_dataset['train']))]
                if wrong_example[field_key] != example[field_key] and wrong_example[field_key] not in wrong_answers:
                    wrong_answers.append(wrong_example[field_key])
            choices = wrong_answers + [example[field_key]]
            out["answer_text"] = example[field_key]

        np.random.shuffle(choices)
        out["choices1"] = choices[0]
        out["choices2"] = choices[1]
        out["choices3"] = choices[2]
        out["choices4"] = choices[3]
        answer_index = choices.index(out["answer_text"]) 
        out["answer"] = '(' + chr(ord('A') + answer_index) + ')'
        return out
    
    # tokenize the dataset
    split_dataset = split_dataset.map(
        process,
        remove_columns=['id', 'age', 'gender', 'location', 'occupation_category', 'annual_wage', 'qualitative_descriptions', 'demographic_summary', 'background_story', 'daily_life', 'digital_behavior', 'financial_situation', 'values_and_beliefs', 'challenges', 'aspirations', 'family_and_relationships', 'personality', 'political_beliefs', 'education'],
        desc="generate choice dataset",
        num_proc=num_proc,
    )
    print(split_dataset)
    print(split_dataset['train'][1])
    print(split_dataset['train'][2])

    
    with open(key_file, 'r') as f:
        hf_key = f.read().strip()
    huggingface_hub.login(token=hf_key)
    split_dataset.push_to_hub(output_path)


# /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python data/phys_llm_31/bio_qa_dataset.py

if __name__ == "__main__":

    for field in ["age", "location", "occupation", "wage",]: #"gender"]:
        output_path = f"synthetic-humans-1m-choice-{field}"
        gen_qa_data(output_path)