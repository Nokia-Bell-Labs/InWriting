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
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
#model_name = "google/gemma-2-9b-it"
#model_name = "Qwen/Qwen3-8B"
hf_token = ####################################
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
        "Act as a fiannce expert and assign the content based to the valid category\n"
        "All possible valid category for you to choose from are as follows (one category per line, in the format of <category>):\n"
        "- Finance\n"
        "- Technology\n"
        "- Tax and Accounting\n"
        "- Business and Management\n"
        "- Government and Controls\n"
        "- Industry\n"
        "Your answer MUST based on the above options, do not answer Insufficient information\n\n\n"
    ),

    # variation 2
    (
        "Follow the instruction to complete the task:\n"
        "Act as a fiannce expert and assign the content based to the valid category\n"
        "Your answer MUST based on the above options, do not answer Insufficient information\n"
        "All possible valid category for you to choose from are as follows (one category per line, in the format of <category>):\n"
        "- Finance\n"
        "- Technology\n"
        "- Tax and Accounting\n"
        "- Business and Management\n"
        "- Government and Controls\n"
        "- Industry    \n\n\n"
    ),

    # variation 3
    (
        "Follow the instruction to complete the task:\n"
        "Act as a fiannce expert and assign the content based to the valid category\n"
        "All possible valid category for you to choose from are as follows (one category per line, in the format of <category>):\n"
        "- Finance\n"
        "- Technology\n"
        "- Tax and Accounting\n"
        "- Business and Management\n"
        "- Government and Controls\n"
        "- Industry\n"
        "Your answer MUST based on the above options, do not answer Insufficient information\n\n\n"
    ),
]

format_descriptions  = [
    # Original format descriptions
    # variation 1
    (
        "Instruct : Derive the most likely category to answer key.\n"
        "Provide your output in the following valid text format:\n"
        "Answer: ...\n\n\n\n"
    ),

    # variation 2
    (
        "Instruct : Derive the most likely category to answer key.\n"
        "Provide your output in the following valid text format:\n"
        "Final Answer: <valid category>\n\n\n\n"
    ),

    # variation 3
    (
        "Instruct : Derive the most likely category to answer key.\n"
        "Provide your output in the following valid text format:\n"
        "Answer: <valid category>\n\n\n\n"
    ),
]

def prompt_multifin(sentence: str, task_var: int, format_var: int):
    # Retrieve task + format templates
    task_description = task_descriptions[task_var]
    format_description = format_descriptions[format_var]

    # Build the full prompt block
    full_prompt = task_description + format_description
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
class multifin(BaseModel):
    final_answer: Literal[
        "Finance",
        "Technology",
        "Tax and Accounting",
        "Business and Management",
        "Government and Controls",
        "Industry"
    ]


processor = SchemaProcessor(
    response_format=multifin,
    tokenizer=tokenizer,
    include_tool_call=False,
    allow_preamble=True,
    max_preamble_tokens=512,
    trigger_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    )

# ------------------------
# DATASET
# ------------------------
dataset = load_dataset('ChanceFocus/flare-multifin-en',split='test')
# ------------------------
# MAIN EVALUATION FUNCTION
# ------------------------
def multifin_run(
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

            sentence = example['text']
            gold_answer = example['answer'].replace('&', 'and')

            messages = prompt_multifin(sentence, task_var, format_var)

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
        multifin_run(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            processor=[processor],
            do_sample=False,
            device=device,
            task_var=task,
            format_var=fmt,
        )

