# AMuRD: Annotated Multilingual Receipts Dataset for  Key Information Extraction and Classification 






## Abstract 

> The extraction of key information from receipts is a complex task that involves the recognition and extraction of text from scanned receipts. This process is crucial as it enables the retrieval of essential content and organizing it into structured documents for easy access and analysis. In this paper, we present AMuRD, a novel multilingual human-annotated dataset specifically designed for information extraction from receipts. This dataset comprises $47,720$ samples and addresses the key challenges in information extraction and item classification - the two critical aspects of data analysis in the retail industry. Each sample includes annotations for item names and attributes such as price, brand, and more. This detailed annotation facilitates a comprehensive understanding of each item on the receipt. Furthermore, the dataset provides classification into $44$ distinct product categories. This classification feature allows for a more organized and efficient analysis of the items, enhancing the usability of the dataset for various applications. In our study, we evaluated various language model architectures, e.g., by fine-tuning LLaMA models on the AMuRD dataset. Our approach yielded exceptional results, with an F1 score of 97.43\% and accuracy of 94.99\% in information extraction and classification, and an even higher F1 score of 98.51\% and accuracy of 97.06\% observed in specific tasks.


## Demo for our Instruct LLama


Explore our Instruct LLama system through our live demo:

[**Demo for our Instruct LLama**](http://3.145.70.14:5003/)


## Examples

| Example | Input                                | Class                 | Brand        | Weight    | Number of units | Size of units  | Price   | T.Price | Pack   | Unit  |
| ------- | ------------------------------------ | ---------------------- | -------------| --------- | ---------------- | --------------- | ------- | ------- | ------ | ----- |
| Example 1| `40.99 20.99 2 chunks sunshine`    | Tins, Jars & Packets   | sunshine     | No Weight | 2                | No Size of units| 20.99   | 40.99   | علبة   | No Unit |
| Example 2| `برسيل اتوماتيك جل روز 2.6`      | Cleaning Supplies      | برسيل       | 2.6ل      | 1                | No Size of units| No Price| No T.Price | عبوة | ل     |
| Example 3| `regina Pasta penne 400g`           | Rice, Pasta & Pulses   | regina       | 400g      | 1                | No Size of units| No Price| No T.Price | كيس   | g     |
| Example 4| `10.00 400g Penne Pasta ElMaleka`   | Rice, Pasta & Pulses   | ElMaleka     | 400g      | 1                | No Size of units| 10      | 10      | كيس   | g     |


## Getting the code


To get started with the code and utilize the AMuRD dataset for your research or projects, you can clone this repository:

### To use our original model on GPU
```bash
git clone https://github.com/Update-For-Integrated-Business-AI/AMuRD.git
pip install transformers
python test.py
```
To perform key information extraction on item receipts using our models, you can utilize the following pre-trained models:

[abdoelsayed/llama-7b-v2-Receipt-Key-Extraction](https://huggingface.co/abdoelsayed/llama-7b-v2-Receipt-Key-Extraction)

[abdoelsayed/llama-7b-v1-Receipt-Key-Extraction](https://huggingface.co/abdoelsayed/llama-7b-v1-Receipt-Key-Extraction)

### To use our original model on the CPU
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
# convert the 7B model to ggml FP16 format
python convert.py path/to/our/model

# quantize the model to 4-bits (using q4_0 method)
./quantize ./path/to/our/model/ggml-model-f32.bin ./path/to/our/model/ggml-model-q4_0.bin q4_0
```

To use the ```ggml-model-q4_0.bin``` on your own desktop, laptop, or low-resource AWS instance, you can follow these general steps:
```

pip install llama-cpp-python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="./models/brand_arabic_models/test/ggml-model-q4_0.bin")
args = parser.parse_args()

llm = Llama(model_path=args.model)

output = llm(
    # SeaStar Smoked herrings fillet, 250 gm
    """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.\n\n
### Instruction:\n Extract the Brand Name from the following Sentence \n\n### Input:\n   SeaStar Smoked herrings fillet, 250 gm  \n\n### Response: """,
    max_tokens=128,
    stop=["Q:", "\n"],
    echo=True,
)

print(output)
```
## Download Links
- **Training Set**: [Download](https://drive.google.com/file/d/1DNdNBnN-i1z6mthrtO6JMtU3pADTNxut/view?usp=sharing)
- **Validation Set**: [Download](https://drive.google.com/file/d/1j8ybqbfMCKEghppfa88eoHAuEvtZ5-dO/view?usp=sharing)
- **Test Set**: [Download](https://drive.google.com/file/d/1l8iUlFnNehIIvkk_xEpB8NDidGX9EF5d/view?usp=sharing)
## Result
 -  LLama v1 Fine tuned [Download](https://github.com/Update-For-Integrated-Business-AI/AMuRD/blob/main/results.zip)
 -  LLama v2 Fine tuned [Download](https://github.com/Update-For-Integrated-Business-AI/AMuRD/blob/main/results.zip)
## License

Note: The AMuRD Dataset can only be used for non-commercial research purposes. 

