import torch
from datasets import load_dataset
from tqdm import tqdm
import json
import time
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import re
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig
import time
from pydantic import BaseModel, Field, ValidationError
from typing import Annotated, Type, Optional
from litelines.transformers import SchemaProcessor

# ======================================================
# Device setup
# ======================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
assert device == torch.device("cuda"), "In Runtime, Change runtime type to GPU"

# ======================================================
# Load models and tokenizers
# ======================================================
#model_name = "Qwen/Qwen3-1.7B"   # primary LLM
#model_name = "Qwen/Qwen3-4B"          # Qwen3 4B
#model_name = "Qwen/Qwen3-8B"          # Qwen3 8B

#model_name = "Qwen/Qwen3.5-2B"
#model_name = "Qwen/Qwen3.5-4B"
#model_name = "Qwen/Qwen3.5-9B"

model_name = "HuggingFaceTB/SmolLM3-3B"

#model_name = "allenai/Olmo-3-7B-Instruct"


#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"   # Llama 3 8B Instruct
#model_name = "google/gemma-2-9b-it"   # Gemma 2 9B Instruct

hf_token = ########################################

def load_main_model(model_name: str, hf_token: str):
    """
    Load the primary LLM with FP16 and automatic device mapping.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )

    print(f"Main model loaded successfully: {model_name}")
    return tokenizer, model

def load_parser_model(parser_name: str, hf_token: str):
    """
    Load the parser model in 4-bit quantization using BitsAndBytes.
    """
    tokenizer = AutoTokenizer.from_pretrained(parser_name, token=hf_token)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        parser_name,
        quantization_config=quant_config,
        device_map="auto",
        token=hf_token
    )

    # Handle padding tokens
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Parser model loaded successfully: {parser_name}")
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


special_token_ids = set(tokenizer.all_special_ids)
pad_id = tokenizer.pad_token_id


# ======================================================
# Prompt builder
# ======================================================

task_descriptions = [
    # variation 1
    (
        "Follow the instruction to complete the task:\n"
        "You are given a string of words and you need to take the last letter of each words and concate them.\n"
        "Read the last question carefully and think step by step before answering. \n\n\n"
    ),

    # variation 2
    (
        "Follow the instruction to complete the task:\n"
        "Read carefully for each of the last question and think step by step before answering. "
        "You are given a string of words and you need to take the last letter of each words and concate them\n\n\n"
    ),

    # variation 3
    (
        "Follow the instruction to complete the task:\n"
        "String manipulation task:\n"
        "• Given: A sequence of words\n"
        "• Required: A new string made from the last letter of each word\n"
        "• Process: Think step by step to solve this challenge\n"
        "Note: Ensure you've read the question thoroughly before beginning.\n\n\n"
    )
]

format_descriptions  = [
    # variation 1
    (
        "Instruct : Provide your output in the following text format:\n"
        "Answer: <reasoning first>. The final answer is <answer>\n\n\n\n"
    ),

    # variation 2
    (
        "Instruct : Provide your output in the following text format:\n"
        "Step by step reasoning: ... \n"
        "Answer: The final answer is ...\n\n\n\n"
    ),

    # variation 3
    (
        "Instruct : Provide your output in the following text format:\n"
        "Answer: <think step by step>. The final answer is <answer>\n\n\n\n"

    ),
]


format_descriptions_BF = [
    (
        "Instruct: First provide your reasoning, then output the final answer strictly as JSON with key 'final_answer'.\n"
        "Format:\n"
        "Answer: <reasoning>\n"
        "{\"final_answer\": \"<final answer>\"}\n"
    ),
]

dspy_prompt = "Your input fields are:\n" \
"1. `question` (str):\n" \
"Your output fields are:\n" \
"1. `reasoning` (str): \n" \
"2. `answer` (str):\n" \
"All interactions will be structured in the following way, with the appropriate values filled in.\n\n" \
"[[ ## question ## ]]\n" \
"{question}\n\n" \
"[[ ## reasoning ## ]]\n" \
"{reasoning}\n\n" \
"[[ ## answer ## ]]\n" \
"{answer}\n\n" \
"[[ ## completed ## ]]\n" \
"In adhering to this structure, your objective is: \n" \
"        Given the fields `question`, produce the fields `answer`."


few_shot_examples = [
    "Question: Take the last letters of the words in \"Elon Musk\" and concatenate them.",
    "Question: Take the last letters of the words in \"Larry Page\" and concatenate them.",
    "Question: Take the last letters of the words in \"Sergey Brin\" and concatenate them.",
    "Question: Take the last letters of the words in \"Bill Gates\" and concatenate them."
]

few_shot_answers = [
    "Answer: The last letter of \"Elon\" is \"n\". The last letter of \"Musk\" is \"k\". Concatenating them is \"nk\". The answer is nk.",
    "Answer: The last letter of \"Larry\" is \"y\". The last letter of \"Page\" is \"e\". Concatenating them is \"ye\". The answer is ye.",
    "Answer: The last letter of \"Sergey\" is \"y\". The last letter of \"Brin\" is \"n\". Concatenating them is \"yn\". The answer is yn.",
    "Answer: The last letter of \"Bill\" is \"l\". The last letter of \"Gates\" is \"s\". Concatenating them is \"ls\". The answer is ls."
]

few_shot_answers_BF = [
    "Answer: The last letter of \"Elon\" is \"n\". The last letter of \"Musk\" is \"k\". Concatenating them is \"nk\". The answer is {\"final_answer\": \"nk\"}.",
    "Answer: The last letter of \"Larry\" is \"y\". The last letter of \"Page\" is \"e\". Concatenating them is \"ye\". The answer is {\"final_answer\": \"ye\"}.",
    "Answer: The last letter of \"Sergey\" is \"y\". The last letter of \"Brin\" is \"n\". Concatenating them is \"yn\". The answer is {\"final_answer\": \"yn\"}.",
    "Answer: The last letter of \"Bill\" is \"l\". The last letter of \"Gates\" is \"s\". Concatenating them is \"ls\". The answer is {\"final_answer\": \"ls\"}."
]

def prompt(sentence: str, task_var: int, format_var: int, n_shots: int, instruct_prompt_format: bool, BF=False, dspy=False):
    # Build system block
    if dspy:
        # If dspy is True, use the dspy prompt
        task_description = task_descriptions[task_var]
        full_prompt = task_description + dspy_prompt
    else:
        # Otherwise use the default task + format descriptions
        task_description = task_descriptions[task_var]
        if not BF: # Use original format variations
            format_description = format_descriptions[format_var]
        else:  # Use Better Format variations
            format_description = format_descriptions_BF[format_var]
        full_prompt = task_description + format_description
    text_sample = f"Question: {sentence}"

    examples = few_shot_examples
    answers = few_shot_answers
    # If instruct_prompt_format = False → everything as user content
    if not instruct_prompt_format:     
        # Build examples block if n_shots > 0
        examples_block = ""
        if n_shots > 0:
            examples_block = "Here are some examples:\n\n"
            for i in range(n_shots):
                examples_block += examples[i].strip()+ "\n" + answers[i].strip() + "\n"

        messages = [
            {
                "role": "user",
                "content": f"{full_prompt}{examples_block}{text_sample}"
            }
        ]
        return messages

    # If instruct_prompt_format = True → structured system + examples + user
    messages = [
        {
            "role": "system",
            "content": full_prompt
        }
    ]

    # Add examples if n_shots > 0
    if n_shots > 0:
        if BF: # Better format variations of fewshots examples
            examples = few_shot_examples
            answers = few_shot_answers_BF

        # Add each example as a pair of user + assistant messages
        for i in range(n_shots):
            messages.append({
                "role": "user",
                "content": examples[i].strip()
            })
            messages.append({
                "role": "assistant",
                "content": answers[i].strip()
            })

    # Add the new user question
    messages.append({
        "role": "user",
        "content": text_sample
    })
    messages.append({
        "role": "assistant",
        "content": ""
    })

    return messages

def parser_prompt(response:str):

    parser_prompt = (
        "Extract the following response final answer, only alphabet from a-z only.\n"
        "DO NOT OUTPUT ANYTHING ELSE OTHER THAN THE FINAL ANSWER!\n"
        "Remove any block elements like <answer> or anything that wasn't the actual lower letter answer\n"
        "Response:\n"
    )

    full_prompt = parser_prompt + response

    messages = [
        {"role": "user", "content": full_prompt}
    ]

    return messages

# ======================================================
# Load dataset
# ======================================================
data = load_dataset("ChilleD/LastLetterConcat", split="test")

# ======================================================
# Define schema for structured response
# ======================================================
class schema(BaseModel):
    #think_step_by_step: str = Field(description="Step-by-step reasoning used to extract the final answer.")
    final_answer: Annotated[
        str,
        Field(
            pattern=r'^[A-Za-z]{4}$',
            description=(
                "You store a string formed by concatenating all extracted last letters here, "
                "keeping the same order."
            )
        )
    ]
def processor(response_format, tokenizer):
    return SchemaProcessor(
        response_format=response_format,
        tokenizer=tokenizer,
        #max_same_state=2,
        include_tool_call=False,
        allow_preamble=True,
        #max_preamble_tokens=512,
        #trigger_token_ids = [tokenizer.eos_token_id]
        #trigger_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
        #trigger_token_ids=[151645] #eos token
    )

processor = processor(schema, tokenizer)
# ======================================================
# Run predictions
# ======================================================

def extract_json_object(text: str, model: Type[BaseModel]) -> Optional[BaseModel]:
    """
    Extracts the FIRST JSON object in text that matches the given Pydantic model.
    """

    json_candidates = re.findall(r"\{.*?\}", text, re.DOTALL)

    for candidate in json_candidates:
        try:
            data = json.loads(candidate.lower())
            return model.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            continue
    return None

def run(
    dataset,
    model,
    tokenizer,
    parser_model=None,
    parser_tokenizer=None,
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
    NL2Format=True,
    BF=False,
    dspy=False,
    batch_size = 1
):

    if processor is None or NL2Format==True:
        processor = []

    if tool is None:
        tool = []

    correct_count = 0
    parsing_fail_count = 0

    for seed in range(start_seed, end_seed + 1):
        set_seed(seed)
        #seed_start_time = time.time()

        # Prepare examples/answers according to n_shots
        if n_shots == 0:
            examples_subset = None
            answers_subset = None
        else:
            if not BF:
                examples_subset = few_shot_examples[:n_shots]
                answers_subset = few_shot_answers[:n_shots]
            else:
                examples_subset = few_shot_examples[:n_shots]
                answers_subset = few_shot_answers_BF[:n_shots]
        if dspy:
            task_var = None
            format_var = None
            dsp_prompt_value = dspy_prompt  # use the DSPy prompt
        else:
            task_var = task_var
            format_var = format_var
            dsp_prompt_value = dspy

        config_json = {
            "task_var": None if task_var is None else task_var+1,
            "task_description": None if task_var is None else task_descriptions[task_var],
            "format_var": None if format_var is None else format_var+1,
            "format_description": (
                None if format_var is None
                else format_descriptions[format_var] if not BF
                else format_descriptions_BF[format_var]
            ),
            "NL-to-Format": NL2Format,
            "n_shots": n_shots,
            "examples": examples_subset,
            "answers": answers_subset,
            "instruct_prompt_format": instruct_prompt_format,
            "DSPy prompt": dsp_prompt_value
            }

        print(json.dumps({"config": config_json}, indent=2))

        parsed_outputs = []
        all_generated_texts_step1 = []
        all_generated_texts_step2 = []
        all_new_tokens_step1 = [] # The step where LLMs generate a answer for given questions
        all_new_tokens_step2 = []
        all_gold_answers = []

        for start in tqdm(range(0, len(dataset["question"]), batch_size), desc=f"Seed {seed}"):
            end = start + batch_size
            batch_examples = dataset["question"][start:end]
            batch_answers = dataset["answer"][start:end]
            all_gold_answers.extend(batch_answers)
            
            messages_batch = [
                prompt(
                    example,
                    task_var,
                    format_var,
                    n_shots,
                    instruct_prompt_format,
                    BF,
                    dspy
                )
                for example in batch_examples
            ]

            formatted_messages = [
                tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tools=tool,
                    enable_thinking=False,
                    tokenize=False                  )
                for messages in messages_batch
            ]

            tokenizer.padding_side = "left"

            inputs = tokenizer(
                formatted_messages,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                logits_processor=processor,
                max_new_tokens=600,
                do_sample=do_sample,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id
            )
    
            generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
            generated_texts = tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            
            all_generated_texts_step1.extend(generated_texts)

            new_tokens_counts = [
                sum(1 for t in row if t.item() not in special_token_ids)
                for row in generated_tokens
            ]
            all_new_tokens_step1.extend(new_tokens_counts)

            #step1_time_sec = round(time.time() - batch_start_time, 2)

            if not NL2Format: # ie Our method
                # ------------------------
                # PARSE JSON RESPONSE
                # ------------------------
                for i, text in enumerate(generated_texts):
                    parsed = extract_json_object(text, schema)
                    if parsed is not None:
                        parsed = parsed.final_answer
                    parsed_outputs.append(parsed)

            else:
                for step1_text in generated_texts:
                    inputs_parser = parser_tokenizer.apply_chat_template(
                        parser_prompt(step1_text),
                        add_generation_prompt=True,
                        tools=tool,
                        enable_thinking=False,
                        tokenize=True,
                        return_tensors="pt"
                    ).to(device)

                    input_parser_length = inputs_parser["input_ids"].shape[1]

                    step2_generated = parser_model.generate(
                        **inputs_parser,
                        max_new_tokens=600,
                        do_sample=do_sample,
                        temperature=0.0
                    )

                    step2_text = parser_tokenizer.decode(
                        step2_generated[0][input_parser_length:],
                        skip_special_tokens=True
                    )

                    all_generated_texts_step2.append(step2_text)
                    parsed_outputs.append(step2_text)
                    all_new_tokens_step2.append(step2_generated[0].shape[0] - input_parser_length)

                
        for j in range(len(parsed_outputs)):
            sample_json = {
                "sample_id": j,
                "sample_question": dataset["question"][j],
                "gold_answer": dataset["answer"][j],
                "model_output_step1": all_generated_texts_step1[j],
                "model_output_step2": all_generated_texts_step2[j] if NL2Format else None,
                "parsed_answer": parsed_outputs[j],
                "new_generated_tokens_step1": all_new_tokens_step1[j],
                "new_generated_tokens_step2": all_new_tokens_step2[j] if NL2Format else 0,
                "new_generated_tokens_total": all_new_tokens_step1[j] +
                                              (all_new_tokens_step2[j] if NL2Format else 0),
            }
            print(json.dumps(sample_json, indent=1))

        correct = [
            (parsed_outputs[i] == all_gold_answers[i])
            for i in range(len(all_gold_answers))
        ]
        accuracy = sum(correct) / len(correct)

        avg_new_tokens_total = sum(
            all_new_tokens_step1[i] +
            (all_new_tokens_step2[i] if NL2Format else 0)
            for i in range(len(parsed_outputs))
        ) / len(parsed_outputs)
        
        fail_parse = sum(x is None for x in parsed_outputs) / len(parsed_outputs)
        print(
            f"fmt={format_var+1}, task={task_var+1}, n_shots={n_shots}, "
            f"accuracy={accuracy:.4f}, fail_parse={fail_parse:.4f}, "
            f"avg_new_tokens_total={avg_new_tokens_total:.2f}"
        )

# ------------------------
# RUN
# ------------------------
def experiment(text, batch_size=1):
    if text.lower() == "lmsf":
        print("\n" + "="*80 + "\n")
        print("Repeat 'Let Me Speak Freely' experiments with Qwen3 32B 4bit quantized\n")
        print("="*80 + "\n\n\n")
        parser_tokenizer, parser_model = load_parser_model(
            "Qwen/Qwen3-32B",
            hf_token
        )

        n_shots_values = [0, 1, 4]
        for task in range(3):
            for fmt in range(3):
                for n_shots in n_shots_values:
                    run(
                        dataset=data,
                        model=model,
                        tokenizer=tokenizer,
                        parser_model=parser_model,
                        parser_tokenizer=parser_tokenizer,
                        do_sample=False,
                        device=device,
                        task_var=task,
                        format_var=fmt,
                        n_shots=n_shots,
                        instruct_prompt_format=False,
                        NL2Format=True,
                        batch_size=batch_size
                    )

    elif text.lower() == "base":
        print("\n" + "="*80 + "\n")
        print("Experiment with Litelines - Base\n")
        print("="*80 + "\n\n\n")

        n_shots_values = [0, 1, 4]

        for task in range(3):
            for fmt in range(3):
                for n_shots in n_shots_values:
                    run(
                        dataset=data,
                        model=model,
                        tokenizer=tokenizer,
                        processor=[processor],
                        do_sample=False,
                        device=device,
                        task_var=task,
                        format_var=fmt,
                        n_shots=n_shots,
                        instruct_prompt_format=False,
                        NL2Format=False,
                        batch_size=batch_size
                    )

    elif text.lower() == "if":
        print("\n" + "="*80 + "\n")
        print("Experiment with Litelines — Instruction Prompt Format\n")
        print("="*80 + "\n\n\n")

        #n_shots_values = [0, 1, 4]
        n_shots_values=[0]

        for task in range(3):
            for fmt in range(3):
                for n_shots in n_shots_values:
                    run(
                        dataset=data,
                        model=model,
                        tokenizer=tokenizer,
                        processor=[processor],
                        do_sample=False,
                        device=device,
                        task_var=task,
                        format_var=fmt,
                        n_shots=n_shots,
                        instruct_prompt_format=True,
                        NL2Format=False,
                        batch_size=batch_size
                    )

    elif text.lower() == "bf":
        print("\n" + "="*80 + "\n")
        print("Replicating 'Let Me Speak Freely' experiment — Better Format\n")
        print("="*80 + "\n\n\n")

        #n_shots_values = [0, 1, 4]
        n_shots_values=[0]

        for task in range(3):
            for fmt in range(1):
                for n_shots in n_shots_values:
                    run(
                        dataset=data,
                        model=model,
                        tokenizer=tokenizer,
                        processor=[processor],
                        do_sample=False,
                        device=device,
                        task_var=task,
                        format_var=fmt,
                        n_shots=n_shots,
                        instruct_prompt_format=True,
                        NL2Format=False,
                        BF=True,
                        batch_size=batch_size
                    )

    else:
        print("Unknown experiment type")

    """
    print("\n" + "="*80 + "\n")
    print(f"Replicating 'Let Me Speak Freely' experiment with Litelines — DSPy prompt\n")
    print("="*80 + "\n\n\n")

    n_shots_values = [0, 1, 4]

    for n_shots in n_shots_values:
        run(
            dataset=data,
            model=model,
            tokenizer=tokenizer,
            processor=[processor],
            do_sample=False,
            device=device,
            task_var=1,
            format_var=1,
            n_shots=n_shots,
            instruct_prompt_format=True,
            NL2Format=False,
            dspy=True
        )
    """
if __name__ == "__main__":
    experiment("base", batch_size=16)
