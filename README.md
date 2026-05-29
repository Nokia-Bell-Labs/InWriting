# Thinking Before Constraining: A Unified Decoding Framework for Large Language Models

This repository contains the implementation of the paper *[Thinking Before Constraining: A Unified Decoding Framework for Large Language Models](https://arxiv.org/abs/2601.07525)*.

The work proposes a hybrid decoding strategy that combines the flexibility of natural language generation with the reliability of structured outputs by decoupling reasoning from formatting.

## Environment Setup

```
pip install -r requirements.txt
```


You need a Hugging Face access token to run properly.

Replace the placeholder in the .py files with your own token:

```
hf_token = ###########

```
## Reproduce Results

Set up the arguments for your desired experiment in the task-dependent .py file.

### InWriting Reproduction: 

For each task, ensure the following configurations are set in your execution script (.py)

Schema: From the task class schema, set only ```final_answer``` as the extractor variable.

Processor: Set the trigger tokens to match the ```EOS``` and ```PAD``` tokens:
```
trigger_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
```

Execution: Call the experiment function with the base configuration and a batch size of 16:
```
experiment("base", batch_size=16)
```

Once the configurations are set, execute the script from your terminal:
```
> python task_name.py
```

## Reference
If you find our work helpful, please cite as:
```
bibtex
@article{nguyen2026thinking,
  title={Thinking Before Constraining: A Unified Decoding Framework for Large Language Models},
  author={Nguyen, Ngoc Trinh Hung and Silva, Alonso and Zumot, Laith and Tupikina, Liubov and Aghasaryan, Armen and Alam, Mehwish},
  journal={arXiv preprint arXiv:2601.07525},
  year={2026}
}
```

