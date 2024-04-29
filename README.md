# AI Judge using Large Language Model

## Three AI Judges trained using QLoRA
- Knowledgeable Taiwanese Judge: './TaiwanLlama'
- Knowledgeable Chinese   Judge: './Llama2-Chinese-7b-Chat'
- Knowledgeable Taiwanese Judge 2: './TaiwanLlama_bad'


## Sources of data:
- https://www.judicial.gov.tw/tw/np-117-1.html （司法院）
- https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjHgM\_5qZmDAxVwjK8BHQvsB-gQFnoECA8QAQ&url=https%3A%2F%2Fwww.grjh.ntpc.edu.tw%2Fapp%2Findex.php%3FAction%3Ddownloadfile%26file%3DWVhSMFlXTm9Mek15TDNCMFlWOHlPVGt6WHpreU16TTVNalZmTlRVNU1qY3VaRzlq%26fname%3DWW54RPOKNPSSTXKK44VWROWTWWNK14KK203435NKIH25ML3134TSA0POFGOOFGGHUS54WWMPA40441JGYWJGA0YSECRKNO3501YTXWB5ROA434SWECOKXSXXYWXW4521JCLKSXIGXSJC24WSUS30A110&usg=AOvVaw3pa0gilcNiTjrqHMA6HzSW&opi=89978449 （高中題目）
- 公務人員特種考試司法人員考試、移民行政考試、調查局、海岸巡防人員考試：刑法、刑法概要
URL:https://www.public.com.tw/exampoint/2022-judicial
- 無罪：Created by human: About 50 data.

## Evaluate
We adopted human evaluation on the testing data to evaluate the preformance of each judges. Two stages evaluations are provided.
- Stage 1: 'Correctness'. The score here is based on the correctness of predicted crime name.
- Stage 2: 'Celebrity Traits'. The score here is based on the voting in our survey, which includes the prediction in testing data for every judges.

