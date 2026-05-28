import os
import json
import torch
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score
from pydantic import BaseModel, Field
from typing import List, Optional, Annotated, Literal
from openai import pydantic_function_tool
from litelines.transformers import SchemaProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
import time
import string


# ======================
# Device setup
# ======================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

assert device == torch.device("cuda"), "In Runtime, Change runtime type to GPU"


# ======================
# Load model and tokenizer
# ======================
#model_name = "Qwen/Qwen2.5-1.5B-Instruct"
#model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
#model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
#model_name = "meta-llama/Llama-3.1-8B-Instruct"
#model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
#model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
#model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
#model_name = "Qwen/QwQ-32B"

hf_token = ###############################
def load_main_model(model_name: str, hf_token: str):
    """
    Load the primary LLM with FP16 and automatic device mapping.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True
    )

    print(f"Main model loaded successfully: {model_name}")
    return tokenizer, model


tokenizer, model = load_main_model(
    model_name,
    hf_token
)

if tokenizer.pad_token_id is None:
    print("PAD TOKEN IS NONE")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id


folder_path = "gsm_symbolic"   # folder containing the json files
parsed_data = []

for i in range(100):
    filename = f"{i:04d}.json"
    filepath = os.path.join(folder_path, filename)

    with open(filepath, "r") as f:
        data = json.load(f)
        
        # Keep only the fields you need
        parsed_entry = {
            "question_parsed": data.get("question_parsed"),
            "answer_parsed": data.get("answer_parsed")
        }
        
        parsed_data.append(parsed_entry)

print(f"Loaded {len(parsed_data)} entries")



# ======================
# Prompt
# ======================


examples = """
There are {t} trees in the {g}. {g} workers will plant trees in the {g} today. After they are done, there will be {tf} trees. How many trees did the {g} workers plant today?

Let’s think step by step. Initially, there are {t} trees. After planting, there are {tf} trees. The number of trees planted is <<tf - t>>. The final answer is <<tf - t>>.

If there are {c} cars in the parking lot and {nc} more cars arrive, how many cars are in the parking lot?

Let’s think step by step. Initially, there are {c} cars. {nc} more cars arrive, so the total becomes <<c + nc>>. The final answer is <<c + nc>>.

{p1} had {ch1} {o1} and {p2} had {ch2} {o1}. If they ate {a} {o1}, how many pieces do they have left in total?

Let’s think step by step. Initially, {p1} had {ch1} {o1}, and {p2} had {ch2} {o1}, making a total of <<ch1 + ch2>>. After eating {a} {o1}, the remaining total is <<ch1 + ch2 - a>>. The final answer is <<ch1 + ch2 - a>>.

{p1} had {l1} {o1}. {p1} gave {g} {o1} to {p2}. How many {o1} does {p1} have left?

Let’s think step by step. {p1} started with {l1} {o1}. After giving {g} {o1} to {p2}, {p1} has <<l1 - g>> {o1} left. The final answer is <<l1 - g>>.

{p1} has {t} {o1}. For Christmas, {p1} got {tm} {o1} from {p2} and {td} {o1} from {p3}. How many {o1} does {p1} have now?

Let’s think step by step. {p1} started with {t} {o1}. {p1} received {tm} {o1} from {p2} and {td} {o1} from {p3}. The total is <<t + tm + td>>. The final answer is <<t + tm + td>>.

There were {c} {o1} in the server room. {nc} more {o1} were installed each day, from {d1} to {d2}. How many {o1} are now in the server room?

Let’s think step by step. Initially, there were {c} {o1}. {nc} {o1} were added each day for <<d2 - d1 + 1>> days, which is <<nc * (d2 - d1 + 1)>>. The total is <<c + nc * (d2 - d1 + 1)>>. The final answer is <<c + nc * (d2 - d1 + 1)>>.

{p1} had {gb1} {o1}. On {day1}, {p1} lost {l1} {o1}. On {day2}, {p1} lost {l2} more. How many {o1} does {p1} have at the end of {day2}?

Let’s think step by step. Initially, {p1} had {gb1} {o1}. After losing {l1} {o1} on {day1}, {p1} had <<gb1 - l1>>. After losing {l2} {o1} on {day2}, the total is <<gb1 - l1 - l2>>. The final answer is <<gb1 - l1 - l2>>.

{p1} has ${m}. {p1} bought {q} {o1} for ${p} each. How much money does {p1} have left?

Let’s think step by step. Initially, {p1} had ${m}. {p1} spent <<q * p>> on {q} {o1}. The remaining money is <<m - q * p>>. The final answer is <<m - q * p>>.
"""



def prompt(sentence: str):
    description = """
    You are an expert in solving grade school math tasks. You will be presented with a grade-school math word problem with symbolic variables and be asked to solve it.

    Before answering, you should reason about the problem (using the <reasoning> field in the response described below). Intermediate symbolic expressions generated during reasoning should be wrapped in << >>.

    Then, output the symbolic expression wrapped in << >> that answers the question. The expressions must use numbers as well as the variables defined in the question. You are only allowed to use the following operations: +, -, /, //, %, (), and int().

    You will always respond in the format described below:
    Let’s think step by step. <reasoning> The final answer is <<symbolic expression>>
    
    Then Output the final answer as a symbolic expression in JSON format exactly like this:
    {"final_answer": <symbolic expression>}
    """

    #Then Output the final answer as a symbolic expression in JSON format exactly like this:
    #{"final_answer": <symbolic expression>}

    
    text_sample = f"Question: {sentence}"
    messages = [
        {
            "role": "user",
            "content": f"{description}\n{examples}\n{text_sample}"
        }
    ]
    return messages

# ======================
# Pydantic schema
# ======================

class ExtractGSMSymbolicTask(BaseModel):
    final_answer: str

# ======================
# Schema processor
# ======================
processor = SchemaProcessor(
    response_format=ExtractGSMSymbolicTask,
    tokenizer=tokenizer,
    include_tool_call=False,
    allow_preamble=True,
    max_preamble_tokens=600,
    trigger_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    #trigger_token_ids=[151645] #eos token
    )



def extract_final_answer(text):
    matches = re.findall(r'final_answer"\s*:\s*"([^"]+)', text)

    return matches[-1].lower() if matches else None

# ======================
# Inference loop
# ======================
def task_run(
    dataset,
    model,
    tokenizer,
    processor: list = None,
    tool: list = None,
    do_sample: bool = True,
    device="cuda"
):
    if processor is None:
        processor = []
    if tool is None:
        tool = []

    set_seed(42)
    parsing_fail_count = 0

    for i, example in enumerate(tqdm(dataset, desc="Sample"), start=1):

        sentence = example["question_parsed"]
        gold_answer = example["answer_parsed"]

        messages = prompt(sentence)

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True
        ).to(device)

        input_ids = inputs["input_ids"]

        prompt_length = inputs["input_ids"].shape[1]

        generated = model.generate(
            **inputs,
            logits_processor=[processor],
            max_new_tokens=1000,
            do_sample=do_sample,
            temperature=0.0

        )

        response = tokenizer.decode(
            generated[0][inputs['input_ids'].shape[-1]:-1],
            skip_special_tokens=True    
        )

        response = response.replace("Ġ", " ").replace("Ċ", "\n")
        response = re.sub(r'[\x00-\x1F\x7F]', '', response)
        # ------------------------
        # PARSE JSON RESPONSE
        # ------------------------
        try:
            pred = extract_final_answer(response)
            
            if pred is None:
                parsing_fail_count += 1

        except Exception as e:
            print("Extraction error:", e)
            pred = None
            parsing_fail_count += 1

        print("\n" + "=" * 50)
        print(f" SAMPLE {i}")
        print("=" * 50)

        print("\nGOLD ANSWER:")
        print(gold_answer.lower())

        print("\nPREDICTED ANSWER:")
        print(pred)

    print("\n" + "-" * 50)
    print(f"TOTAL PARSING FAILURES: {parsing_fail_count}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    start_time = time.time()
    task_run(
        dataset=parsed_data,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        do_sample=False,
        device=device
    )





