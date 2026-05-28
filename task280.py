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
#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "google/gemma-2-9b-it"
#model_name = "Qwen/Qwen3-8B"
hf_token = #####################################################
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
task_descriptions = [
    # variation 1
    (
        "Follow the instruction to complete the task:\n"
        "In this task, you are given a short passage that conveys stereotype or anti-stereotype about a specific target. A stereotype is an over-generalized belief about a particular group of people. An anti-stereotype is an idea that goes against a common stereotype. You are expected to classify the passage into four types of stereotype or anti-stereotype: gender, profession, race, and religion.\n\n\n"
    ),

    # variation 2
    (
        "Follow the instruction to complete the task:\n"
        "You are expected to classify the passage into four types of stereotype or anti-stereotype: gender, profession, race, and religion.\n"
        "In this task, you are given a short passage that conveys stereotype or anti-stereotype about a specific target. A stereotype is an over-generalized belief about a particular group of people. An anti-stereotype is an idea that goes against a common stereotype.\n\n\n"
    ),

    # variation 3
    (
        "Follow the instruction to complete the task:\n"
        "Sentence stereotype assessment:\n"
        "\u2022 Given: A passage related to stereotype or anti-stereotype\n"
        "\u2022 Required: Determine if the paragraph is one of these four category : gender, profession, race, and religion\n"
        "\u2022 Output: Answer only one of the four category\n\n\n"
    ),
]

format_descriptions  = [
    # Original format descriptions
    # variation 1
    (
        "Instruct : Provide your output in the following text format, answer in only valid category in string:\n"
        "Answer: ...\n\n\n\n"
    ),

    # variation 2
    (
        "Instruct : Provide your output in the following text format:\n"
        "Step by step reasoning: ...\n"
        "Answer: The final answer is ...\n\n\n\n"
    ),

    # variation 3
    (
        "Instruct : Provide your output in the following text format:\n"
        "Answer: <valid label>\n\n\n\n"
    ),
]

def prompt_task280(sentence: str, task_var: int, format_var: int):
    # Retrieve task + format templates
    task_description = task_descriptions[task_var]
    format_description = format_descriptions[format_var]

    # Build the full prompt block
    full_prompt = task_description + format_description
    text_sample = f"{sentence}"
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
class task280 (BaseModel):
    final_answer: Literal["gender", "profession", "race", "religion"]

processor = SchemaProcessor(
    response_format=task280,
    tokenizer=tokenizer,
    include_tool_call=False,
    allow_preamble=True,
    max_preamble_tokens=512,
    trigger_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]

)

# ------------------------
# DATASET
# ------------------------
with open('data/task280_stereoset_classification_stereotype_type.json', 'r') as f:
    raw_data = json.load(f)
dataset = []
for row in raw_data['Instances'][:1000]:
    dataset.append({
        'question': row['input'],
        'answer': row['output'][0].lower()
    })
# ------------------------
# MAIN EVALUATION FUNCTION
# ------------------------
def task280_run(
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
            "task_var": task_var + 1 ,
            "task_description": task_descriptions[task_var],
            "format_var": format_var + 1,
            "format_description": format_descriptions[format_var],
            "n_shots": n_shots,
            "examples": examples_subset,
            "answers": answers_subset,
            "instruct_prompt_format": instruct_prompt_format
        }
        print(json.dumps({"config": config_json}, indent=2))

        seed_start_time = time.time()  # Start timing for this seed

        for i, example in enumerate(tqdm(dataset, desc=f"Seed {seed}"), start=1):
            sample_start_time = time.time()

            sentence = example['question']
            gold_answer = example['answer']

            messages = prompt_task280(sentence, task_var, format_var)
            
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
        print(f"Time taken for Task {task_var + 1}, Format {format_var + 1}, n_shots {n_shots}: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print("\n\n\n\n\n\n")

# ------------------------
# RUN
# ------------------------

print("\n" + "="*80 + "\n")
print(f"Replicating 'Let Me Speak Freely' experiment with Litelines\n")
print("="*80 + "\n\n\n")

for task in range(3):
    for fmt in range(3):
        task280_run(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            processor=[processor],
            do_sample=False,
            device=device,
            task_var=task,
            format_var=fmt,
        )

