# Knowledgeable Chinese Judge

## Setting
- QLoRA with alpha=8, r=4.
- Prompt: 你現在是法官，我會給你一個事件，你必須要給出判決。USER: {instruction} ASSISTANT:


## Data
- 434 training data from the Internet. (train.json)
- 10  testing data from the Internet. (test.json)

## Change to readble JSON file
```
python3 transform_to_proper_json.py --input [INPUT] -- output [OUPUT]
```

## Train
```
python3 train.py --train_file "train.json" --epochs 3 --save_path [Your Adapter Path] --model_name_or_path "FlagAlpha/Llama2-Chinese-7b-Chat"
```

## Predict
```
python3 predict.py --peft_path "ChineseLlama_final" --valid_file "test.json" --output_file "output.json" --model_name_or_path "FlagAlpha/Llama2-Chinese-7b-Chat"
```


## Demo
```
python3 demo.py --peft_path "ChineseLlama_final" --model_name_or_path "FlagAlpha/Llama2-Chinese-7b-Chat"
```

# Model Version
- ChineseLlama\_final: Training with 434 training data with unified output format.
- ChineseLlama\_370\_revise: Training with 370 training data with unified output format.
- ChineseLlama\_370\_not\_revise: Training with 370 training data without unified output format.

