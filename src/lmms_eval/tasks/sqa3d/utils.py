import re
import os
import pandas as pd
from pathlib import Path
import yaml
import pickle
from PIL import Image
from loguru import logger as eval_logger

with open(Path(__file__).parent / "sqa3d.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
media_dir = yaml.safe_load("".join(safe_data))["metadata"]["media_dir"]
# embodiedscan_path = yaml.safe_load("".join(safe_data))["metadata"]["embodiedscan_path"]
# with open(embodiedscan_path, "rb") as f:
#     data = pickle.load(f)["data_list"]
#     id2scene = {sample["sample_id"]: sample for sample in data}

def sqa3d_doc_to_visual(doc):
    image_files = doc["images"]
    images = [
        Image.open(
            os.path.join(media_dir, image_file)
        ).convert("RGB")
        for image_file in image_files
    ]
    return [images]

def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


def sqa3d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt")
    question = doc["situation"] + " " + doc["question"]
    return "\n".join([pre_prompt, question, post_prompt])


def sqa3d_process_results(doc, results):
    prediction = clean_answer(results[0].strip().split("\n")[0])
    answer = clean_answer(doc["answer"])
    doc["pred_answer"] = prediction
    doc["result"] = answer == prediction
    return {"sqa3d_score": doc}

def sqa3d_aggregate_results(results):
    results = pd.DataFrame(results)

    def calculate_accuracy(df, source):
        source_df = df[df['question_type'] == source]
        accuracy = source_df['result'].mean()  # Assuming 'result' is 1 for correct and 0 for incorrect
        return accuracy

    output = {}
    for k in ['what', "is", "how", "can", "which", "others"]:
        output[f"{k}_accuracy"] = calculate_accuracy(results, k)
    
    output["overall_accuracy"] = results["result"].mean()
    eval_logger.info(output)

    return output["overall_accuracy"] * 100

