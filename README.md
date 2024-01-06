# MyanmarGPT - Myanmar Generative Pretrained Transformer
 **The very first and largest usable Burmese Language GPT in Myanmar**

- Free to use and open-source
- Lightweight and Accurate
- Burmese + International Languages (Total 61 Languages)
- And, It is Awesome!

MyanmarGPT is the very first and largest Burmese language GPT in Myanmar with strong community contributions. It was created by me, [Min Si Thu](https://www.linkedin.com/in/min-si-thu/).

These two models are trained by using private property datasets, manually clean by Min Si Thu.

There are two versions of MyanmarGPT at the moment, 2023 December.
- [MyanmarGPT](https://huggingface.co/jojo-ai-mst/MyanmarGPT) - 128 M parameters
- [MyanmarGPT-Big](https://huggingface.co/jojo-ai-mst/MyanmarGPT-Big) - 1.42 B parameters

## MyanmarGPT

MyanmarGPT is 128 million parameters Burmese Language Model.
It is very lightweight and easy to use on all devices. 

## MyanmarGPT-Big

MyanmarGPT-Big is 1.42 billion parameters Multi-Language Model.
It is enterprise-level LLM for Burmese Language mainly and other languages.
Currently supports 61 Languages.

## How to use MyanmarGPT and MyanmarGPT-Big

Install huggingface transformer
```shell
pip install transformers
```

### MyanmarGPT

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="jojo-ai-mst/MyanmarGPT")
outputs = pipe("အီတလီ",do_sample=False)
print(outputs)

```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("jojo-ai-mst/MyanmarGPT")
model = AutoModelForCausalLM.from_pretrained("jojo-ai-mst/MyanmarGPT")


input_ids = tokenizer.encode("ချစ်သား", return_tensors='pt')
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```


### MyanmarGPT-Big

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="jojo-ai-mst/MyanmarGPT-Big")
outputs = pipe("အီတလီ",do_sample=False)
print(outputs)

```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("jojo-ai-mst/MyanmarGPT-Big")
model = AutoModelForCausalLM.from_pretrained("jojo-ai-mst/MyanmarGPT-Big")


input_ids = tokenizer.encode("ချစ်သား", return_tensors='pt')
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Contributors

- Min Si Thu
