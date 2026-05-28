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
#model_name = "google/gemma-2-9b-it"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
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
task_descriptions = [
    # Original task_descriptions
    # variation 1
    (
        "Follow the instruction to complete the task:\n"
        "You are given a sentence and your task is to determine whether a sentence relating to sports is plausible or implausible\n"
        "Read carefully for each of the last question and think step by step before answering. \n"
        "Answer yes if its plausible, no if implausible\n\n\n"
    ),

    # variation 2
    (
        "Follow the instruction to complete the task:\n"
        "You are given a sentence and your task is to determine whether a sentence relating to sports is plausible or implausible. Read carefully for each of the last question and think step by step before answering. Answer yes if its plausible, no if implausible\n\n\n"
    ),

    # variation 3
    (
        "Follow the instruction to complete the task:\n"
        "Sentence plausibility assessment:\n"
        "\u2022 Given: A sentence related to sports\n"
        "\u2022 Required: Determine if the sentence is plausible or implausible\n"
        "\u2022 Process: Think step by step to analyze the sentence\n"
        "\u2022 Output: Answer \"yes\" if plausible, \"no\" if implausible\n\n\n"
    ),
]

format_descriptions  = [
    # Original format descriptions
    # variation 1
    (
        "Instruct : Provide your output in the following text format:\n"
        "Answer: ... So the answer is ...\n\n\n\n"
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
        "Answer: <reasoning first>. So the answer is <answer>\n\n\n\n"
    ),
]

def prompt_sports(sentence: str, task_var: int, format_var: int):
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
class sports(BaseModel):
    think_step_by_step: str = Field(description="Step-by-step thinking used to extract the final answer.")
    final_answer: Literal["yes","no"]

processor = SchemaProcessor(
    response_format=sports,
    tokenizer=tokenizer,
    include_tool_call=False,
    allow_preamble=False,
    max_preamble_tokens=512,
    trigger_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    )

# ------------------------
# DATASET
# ------------------------
dataset = load_dataset(
    'tasksource/bigbench',
    'sports_understanding',
    split='validation',
    trust_remote_code=True
)
# ------------------------
# MAIN EVALUATION FUNCTION
# ------------------------
def sports_run(
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

            sentence = example['inputs'].replace('Determine whether the following statement or statements are plausible or implausible:','').replace('Statement: ','').replace('Plausible/implausible?','').strip()
            gold_answer = 'yes' if example['targets'][0] == 'plausible' else 'no'

            messages = prompt_sports(sentence, task_var, format_var)

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
        sports_run(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            processor=[processor],
            do_sample=False,
            device=device,
            task_var=task,
            format_var=fmt,
        )

