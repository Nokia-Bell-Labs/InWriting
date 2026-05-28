import torch
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import json
import time
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, set_seed
import string
from pydantic import BaseModel, Field
from typing import Annotated, List, Literal
from litelines.transformers import SchemaProcessor

# ------------------------
# DEVICE SETTINGS
# ------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

assert device == torch.device("cuda"), "In Runtime, change runtime type to GPU."

# ------------------------
# MODEL LOADING
# ------------------------

#model_name = "Qwen/Qwen3-8B"
model_name = "google/gemma-2-9b-it"
#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
hf_token = #######################################
tokenizer = AutoTokenizer.from_pretrained(model_name,token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name,token=hf_token,torch_dtype=torch.float16).to(device)
if tokenizer.pad_token_id is None:
    print("PAD TOKEN IS NONE")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
print(f"Model loaded successfully: {model_name}")
# ------------------------
# PROMPT FUNCTION
# ------------------------

diseases = [
	"- Possible NSTEMI / STEMI\n",
	"- Spontaneous rib fracture\n",
	"- Pulmonary embolism\n",
	"- Pulmonary neoplasm\n",
	"- URTI\n",
	"- Croup\n",
	"- Sarcoidosis\n",
	"- HIV (initial infection)\n",
	"- Inguinal hernia\n",
	"- Spontaneous pneumothorax\n",
	"- Bronchospasm / acute asthma exacerbation\n",
	"- Viral pharyngitis\n",
	"- Bronchiolitis\n",
	"- Pancreatic neoplasm\n",
	"- Guillain-Barr\u00e9 syndrome\n",
	"- Chagas\n",
	"- Allergic sinusitis\n",
	"- Acute rhinosinusitis\n",
	"- PSVT\n",
	"- Panic attack\n",
	"- Epiglottitis\n",
	"- Bronchiectasis\n",
	"- Bronchitis\n",
	"- Pericarditis\n",
	"- Acute COPD exacerbation / infection\n",
	"- Ebola\n",
	"- Chronic rhinosinusitis\n",
	"- Acute otitis media\n",
	"- Larygospasm\n",
	"- Influenza\n",
	"- Stable angina\n",
	"- Myasthenia gravis\n",
	"- Myocarditis\n",
	"- SLE\n",
	"- GERD\n",	
	"- Anemia\n",
	"- Cluster headache\n",
	"- Localized edema\n",
	"- Anaphylaxis\n",
	"- Atrial fibrillation\n",
	"- Acute pulmonary edema\n",
	"- Acute laryngitis\n",
	"- Acute dystonic reactions\n",
	"- Boerhaave\n",
	"- Pneumonia\n",
	"- Tuberculosis\n",
	"- Whooping cough\n",
	"- Unstable angina\n",
	"- Scombroid food poisoning\n\n\n"
]

task_descriptions = [	
    # variation 2
    (
	"Follow the instruction to complete the task:\n"
	"Act as a medical doctor and diagnose the patient based on the given patient profile\n"
	"All possible valid diagnoses for you to choose from are as follows (one diagnosis per line, in the format of <diagnosis>):\n"
    ),

    # variation 3
    (
	"Follow the instruction to complete the task:\n"
	"Medical diagnosis task:\n"
	"\u2022 Given: A patient profile\n"
	"\u2022 Required: Diagnose the patient based on the provided information\n"
	"\u2022 Process: Think step by step to analyze the patient's symptoms and history\n"
	"\u2022 Output: Select one diagnosis from the provided list of valid options\n"
	"Note: Carefully review the patient profile and the list of possible diagnoses before making your determination. Do not answer \"Insufficient information\" - you must choose from the given options.\n"
	"Valid diagnoses (select one):\n"
    ),
]

format_descriptions  = [
    # Original format descriptions
    # variation 1
    (
	"Instruct : Now, take a deep breath and work on this problem step-by-step to derive the most likely diagnosis. Do not repl\n"
	"Provide your output in the following valid text format:\n"
	"Answer: ...reasoning here... The answer is ...\n\n\n\n"
    ),

    # variation 2
    (
	"Instruct : Now, take a deep breath and work on this problem step-by-step to derive the most likely diagnosis. Do not repl\n"
	"Provide your output in the following valid text format:\n"
	"Step by step reasoning: ...\n"
	"Answer: ...\n\n\n\n"
    ),

    # variation 3
    (
	"Instruct : Now, take a deep breath and work on this problem step-by-step to derive the most likely diagnosis. Do not repl\n"
	"Provide your output in the following valid text format:\n"
	"Answer: [think step by step] The answer is [answer here]\n\n\n\n"
    ),
]

def prompt_ddxplus(sentence: str, task_var: int, format_var: int):
    # Retrieve task + format templates
    task_description = task_descriptions[task_var]
    format_description = format_descriptions[format_var]

    # Build the full prompt block
    diseases_text = "".join(diseases)
    full_prompt = task_description + diseases_text + format_description
    text_sample = f"Question: {sentence}"
    messages = [
        {
            "role": "user",
            "content": f"{full_prompt}{text_sample}"
        }
    ]

    return messages

# ------------------------
# SCHEMA VALIDATION
# ------------------------
class ddxplus(BaseModel):
    final_answer: Literal[
        "Possible NSTEMI / STEMI",
        "Spontaneous rib fracture",
        "Pulmonary embolism",
        "Pulmonary neoplasm",
        "URTI",
        "Croup",
        "Sarcoidosis",
        "HIV (initial infection)",
        "Inguinal hernia",
        "Spontaneous pneumothorax",
        "Bronchospasm / acute asthma exacerbation",
        "Viral pharyngitis",
        "Bronchiolitis",
        "Pancreatic neoplasm",
        "Guillain-Barré syndrome",
        "Chagas",
        "Allergic sinusitis",
        "Acute rhinosinusitis",
        "PSVT",
        "Panic attack",
        "Epiglottitis",
        "Bronchiectasis",
        "Bronchitis",
        "Pericarditis",
        "Acute COPD exacerbation / infection",
        "Ebola",
        "Chronic rhinosinusitis",
        "Acute otitis media",
        "Larygospasm",
        "Influenza",
        "Stable angina",
        "Myasthenia gravis",
        "Myocarditis",
        "SLE",
        "GERD",
        "Anemia",
        "Cluster headache",
        "Localized edema",
        "Anaphylaxis",
        "Atrial fibrillation",
        "Acute pulmonary edema",
        "Acute laryngitis",
        "Acute dystonic reactions",
        "Boerhaave",
        "Pneumonia",
        "Tuberculosis",
        "Whooping cough",
        "Unstable angina",
        "Scombroid food poisoning"
    ]

processor = SchemaProcessor(
    response_format=ddxplus,
    tokenizer=tokenizer,
    include_tool_call=False,
    allow_preamble=True,
    max_preamble_tokens=512,
    trigger_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    )

# ------------------------
# DATASET
# ------------------------
dataset = load_dataset(
    "appier-ai-research/StreamBench",
    "ddxplus",
    split="test",
)
# ------------------------
# MAIN EVALUATION FUNCTION
# ------------------------
def ddxplus_run(
    dataset,
    model,
    tokenizer,
    start_seed: int = 1,
    end_seed: int = 1,
    processor: list = None,
    tool: list = None,
    do_sample: bool = True,
    device=device,
    task_var=0,
    format_var=0,
    n_shots=0,
    instruct_prompt_format=False,
):

    if processor is None:
        processor = []
    if tool is None:
        tool = []

    correct_count = 0
    parsing_fail_count = 0
    
    diseases_text = "".join(diseases)

    for seed in range(start_seed, end_seed + 1):
        set_seed(seed)
        # Prepare examples/answers according to n_shots
        if n_shots == 0:
            examples_subset = None
            answers_subset = None
        else:
            examples_subset = examples[:n_shots]
            answers_subset = answers[:n_shots]
        config_json = {
            "task_var": task_var + 2 ,
            "task_description": task_descriptions[task_var]+ diseases_text,
            "format_var": format_var + 1,
            "format_description": format_descriptions[format_var],
            "n_shots": n_shots,
            "examples": examples_subset,
            "answers": answers_subset,
            "instruct_prompt_format": instruct_prompt_format
        }
        print(json.dumps({"config": config_json}, indent=2))

        seed_start_time = time.time()  

        for i, example in enumerate(tqdm(dataset, desc=f"Seed {seed}"), start=1):
            sample_start_time = time.time()

            sentence = example["PATIENT_PROFILE"]
            gold_answer = example["PATHOLOGY"]

            messages = prompt_ddxplus(sentence, task_var, format_var)

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tools=tool,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True
            ).to(device)

            generated = model.generate(
                **inputs,
                logits_processor=processor,
                max_new_tokens=600,
                do_sample=do_sample,
                temperature=float(0.0)
            )

            # compute token lengths
            generated_tokens = generated[0]
            input_length = inputs["input_ids"].shape[1]
            total_length = generated_tokens.shape[0]
            new_tokens = total_length - input_length

            response = tokenizer.decode(generated[0][inputs['input_ids'].shape[-1]:-1])

            # ------------------------
            # PARSE JSON RESPONSE
            # ------------------------
            try:
                match = re.search(r"\{.*?\}", response, re.DOTALL)
                if match:
                    try:

                        json_obj = json.loads(match.group(0))
                        pred = json_obj.get("final_answer", None)
                    except json.JSONDecodeError:
                        pred = None
                        parsing_fail_count += 1
                else:
                    pred = None
                    parsing_fail_count += 1

            except Exception:
                pred = None
                parsing_fail_count += 1

            correct = (pred == gold_answer)

            if correct:
                correct_count += 1

            sample_json = {
                "sample_id": i,
                "gold_answer": gold_answer,
                "model_output": response,
                "parsed_answer": pred,
                "correct": correct,
                "new_generated_tokens": new_tokens,
                "time_sec": round(time.time() - sample_start_time, 2)
            }
            print(json.dumps(sample_json, indent=1))

        seed_end_time = time.time()
        elapsed_time = seed_end_time - seed_start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        accuracy_so_far = correct_count / len(dataset)
        print(f"Final Accuracy: {accuracy_so_far:.2%}")
        print(f"Final Parsing Failures: {parsing_fail_count}")
        print(f"Time taken for Task {task_var + 2}, Format {format_var + 1}, n_shots {n_shots}: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print("\n\n\n\n\n\n")

# ------------------------
# RUN
# ------------------------

print("\n" + "="*80 + "\n")
print(f"Replicating 'Let Me Speak Freely' experiment with Litelines\n")
print("="*80 + "\n\n\n")

for task in range(2):          
    for fmt in range(3):       
        ddxplus_run(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            processor=[processor], 
            do_sample=False,
            device=device,
            task_var=task,
            format_var=fmt,
        )


