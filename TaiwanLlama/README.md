# Knowledgeable Taiwanese Judge

## Setting
- QLoRA with alpha=8, r=4.
- Prompt: 你現在是法官，我會給你一個事件，你必須要給出判決。USER: {instruction} ASSISTANT:

## Data
- 434 training data from the Internet. (train.json)
- 10  testing  data from the Internet. (test.json)

## Change to readble JSON file
```
python3 transform_to_proper_json.py --input [INPUT] -- output [OUPUT]
```

## Train
```
python3 train.py --train_file "train.json" --epochs 3 --save_path [Your Adapter Path]
```

## Predict
```
python3 predict.py --peft_path "TaiwanLlama_final" --valid_file "test.json" --output_file "output.json"
```


## Demo
```
python3 demo.py --peft_path "TaiwanLlama_final"
```
## Model Version
- TaiwanLlama\_final: Training with 434 training data, with unified data format.
- TaiwanLlama\_200: Training with 200 training data, without unified data format.
