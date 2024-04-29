# AI Judge using Large Language Models

An AI Judge system that autonomously delivers legal judgments for criminal cases using transformer-based Large Language Models (LLM) enhanced with QLoRA tuning.

Developer: Yi-Jie Cheng, Wei-Hsiang Huang, Zhi-Bao Lu, and Chia-Hung Chiang

Course: NTU CSIE 5431 Applied Deep Learning, Fall 2023

Instructor: Yun-Nung (Vivian) Chen

## Overview
This project implements an AI Judge system that autonomously provides legal judgments for criminal cases using transformer-based LLMs. The AI Judge system leverages models like TaiwanLlama and ChineseLlama, enhanced with QLoRA tuning to deliver accurate legal judgments. This system is designed to bridge the gap between complex legal knowledge and public accessibility.


## Two AI Judges trained using QLoRA
- Taiwanese-Judge: [yentinglin/Taiwan-LLM-13B-v2.0-chat](https://huggingface.co/yentinglin/Taiwan-LLM-13B-v2.0-chat)
- Chinese-Judge: [FlagAlpha/Llama2-Chinese-7b-Chat](https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat)


## Sources of data:
- [司法院全球資訊網](https://www.judicial.gov.tw/tw/np-117-1.html)
- [國民中學段考公民科試題](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjHgM\_5qZmDAxVwjK8BHQvsB-gQFnoECA8QAQ&url=https%3A%2F%2Fwww.grjh.ntpc.edu.tw%2Fapp%2Findex.php%3FAction%3Ddownloadfile%26file%3DWVhSMFlXTm9Mek15TDNCMFlWOHlPVGt6WHpreU16TTVNalZmTlRVNU1qY3VaRzlq%26fname%3DWW54RPOKNPSSTXKK44VWROWTWWNK14KK203435NKIH25ML3134TSA0POFGOOFGGHUS54WWMPA40441JGYWJGA0YSECRKNO3501YTXWB5ROA434SWECOKXSXXYWXW4521JCLKSXIGXSJC24WSUS30A110&usg=AOvVaw3pa0gilcNiTjrqHMA6HzSW&opi=89978449)
- [公職王公務人員考試解答](https://www.public.com.tw/exampoint/2022-judicial)
- 無罪：Created by human

