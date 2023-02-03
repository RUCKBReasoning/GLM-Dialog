# GLM-Dialog

This repository stores the codes and references for the paper `GLM-Dialog xxx`

GLM-Dialog is a xxx.

## Model Checkpoints

The trained checkpoint for our final model can be downloaded [here](todo)

## Code

This repository extends on the `SwissArmyTransformer` module, modifications are required to practice our model, we provide the edited version in `./SwissArmyTransformer`

We also provide the cache of web searches during our experiements. A detailed documentation about implementing a Search Engine could be found in [here](doc/replacingSearchEngine.md).

We release the code for our evaluation framework at [DialEvaluation](https://github.com/RUCKBReasoning/DialEvaluation), it can be simply deployed as

```bash
git clone https://github.com/RUCKBReasoning/DialEvaluation
bash setup.sh
bash run.sh
```

## Performances

Our model achieves promising performances as shown below: 

### Human-evaluation on 50 chit-chat self-chat dialogues

| Model      | Coherence    | Informativeness | Safety       | Inspiration  | Hallucination | Engagingness | Faithfulness |
| ---------- | ------------ | --------------- | ------------ | ------------ | ------------- | ------------ | ------------ |
| CDial-GPT  | 0.860        | 0.851           | 0.913        | 0.515        | 0.291         | 0.500        | 0.473        |
| PLATO-XL   | <u>1.455</u> | <u>1.438</u>    | 1.448        | <u>1.129</u> | **0.062**     | <u>1.260</u> | <u>1.220</u> |
| EVA2.0     | 1.386        | 1.336           | 1.362        | 0.902        | <u>0.068</u>  | 1.213        | 1.093        |
| GLM10B     | 1.371        | 1.296           | <u>1.539</u> | 0.932        | 0.130         | 1.187        | 1.160        |
| DialGLM10B | **1.515**    | **1.517**       | **1.656**    | **1.171**    | 0.098         | **1.383**    | **1.383**    |

### Human-evaluation on 100 knowledge-grounded self-chat dialogues

| Model      | Coherence    | Informativeness | Safety       | Inspiration  | Hallucination | Engagingness | Faithfulness |
| ---------- | ------------ | --------------- | ------------ | ------------ | ------------- | ------------ | ------------ |
| CDial-GPT  | 1.140        | 1.069           | 1.478        | 0.591        | 0.221         | 0.603        | 0.690        |
| PLATO-XL   | <u>1.698</u> | <u>1.614</u>    | <u>1.793</u> | 1.090        | **0.032**     | 1.420        | <u>1.413</u> |
| EVA2.0     | 1.488        | 1.413           | 1.674        | 0.832        | 0.089         | 1.230        | 1.223        |
| GLM10B     | 1.513        | 1.497           | 1.669        | <u>1.157</u> | 0.093         | <u>1.460</u> | 1.340        |
| DialGLM10B | **1.759**    | **1.742**       | **1.816**    | **1.223**    | <u>0.046</u>  | **1.550**    | **1.473**    |

### Human-evaluation on 50 chit-chat human-bot chat dialogue

| Model      | Coherence    | Informativeness | Safety       | Inspiration  | Hallucination | Engagingness | Faithfulness |
| ---------- | ------------ | --------------- | ------------ | ------------ | ------------- | ------------ | ------------ |
| CDial-GPT  | 1.138        | 0.984           | 1.310        | 0.690        | 0.272         | 0.696        | 0.660        |
| PLATO-XL   | **1.725**    | <u>1.610</u>    | <u>1.741</u> | 1.239        | **0.068**     | <u>1.392</u> | <u>1.316</u> |
| EVA2.0     | <u>1.690</u> | 1.494           | **1.743**    | 1.107        | <u>0.077</u>  | 1.312        | 1.292        |
| GLM10B     | 1.439        | 1.436           | 1.513        | <u>1.249</u> | 0.164         | 1.236        | 1.208        |
| GLM130B    | 1.232        | 1.179           | 1.378        | 1.000        | 0.257         | 0.816        | 0.784        |
| DialGLM10B | 1.660        | **1.641**       | 1.688        | **1.376**    | 0.127         | **1.440**    | **1.460**    |

### Human-evaluation on 100 knowledge-grounded human-bot chat dialogue

| Model      | Coherence    | Informativeness | Safety       | Inspiration  | Hallucination | Engagingness | Faithfulness |
| ---------- | ------------ | --------------- | ------------ | ------------ | ------------- | ------------ | ------------ |
| CDial-GPT  | 0.956        | 0.777           | 1.194        | 0.543        | 0.363         | 0.562        | 0.542        |
| PLATO-XL   | <u>1.585</u> | 1.387           | <u>1.650</u> | 1.086        | **0.129**     | 1.244        | 1.128        |
| EVA2.0     | 1.524        | 1.275           | 1.616        | 0.961        | 0.151         | 1.150        | 1.096        |
| GLM10B     | 1.543        | <u>1.528</u>    | 1.570        | <u>1.329</u> | 0.174         | <u>1.324</u> | <u>1.282</u> |
| GLM130B    | 1.177        | 1.128           | 1.315        | 0.954        | 0.303         | 0.852        | 0.832        |
| DialGLM10B | **1.668**    | **1.624**       | **1.688**    | **1.393**    | <u>0.134</u>  | **1.412**    | **1.368**    |

## Automatic evaluation results on Diamante

| Model      | Dist-3     | Dist-4     | Bleu-4    | F1         | Rouge-L    | Rouge-1    | Rouge-2    | Bert-Score |
| ---------- | ---------- | ---------- | --------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| CDial-GPT  | 75.300     | 88.584     | 1.936     | 17.791     | 15.108     | 45.732     | 19.550     | 0.603      |
| PLATO-XL   | 73.093     | 85.802     | **2.807** | **18.516** | **16.941** | 53.659     | 21.033     | **0.618**  |
| EVA2.0     | **94.349** | **98.144** | 0.729     | 11.858     | 10.285     | 50.730     | 16.107     | 0.580      |
| GLM10B     | 68.133     | 78.051     | 0.872     | 12.675     | 11.399     | **83.537** | **33.489** | 0.588      |
| GLM130B    | 81.323     | 91.634     | 2.396     | 16.505     | 14.666     | 63.462     | 19.430     | 0.608      |
| DialGLM10B | 85.028     | 94.952     | 1.338     | 13.548     | 12.306     | 69.106     | 23.307     | 0.597      |

### Automatic evaluation results on DuSincR

| Model      | Dist-3        | Dist-4        | Bleu-4       | F1            | Rouge-L       | Rouge-1       | Rouge-2       | Bert-Score   |
| ---------- | ------------- | ------------- | ------------ | ------------- | ------------- | ------------- | ------------- | ------------ |
| CDial-GPT  | 61.477        | 80.521        | 0.792        | 14.652        | 12.011        | 48.212        | 15.707        | 0.580        |
| PLATO-XL   | 26.336        | 40.919        | 1.959        | 16.967        | <u>15.396</u> | 67.397        | 24.011        | 0.607        |
| EVA2.0     | 53.872        | 68.859        | 0.737        | 13.548        | 11.589        | 54.270        | 14.211        | 0.591        |
| GLM10B     | 71.937        | <u>87.969</u> | 2.723        | 15.517        | 12.538        | **83.832**    | **33.743**    | 0.599        |
| GLM130B    | <u>73.604</u> | 87.685        | **4.177**    | **18.905**    | **16.047**    | 79.562        | 28.897        | **0.615**    |
| DialGLM10B | **78.705**    | **93.135**    | <u>3.508</u> | <u>17.478</u> | 14.306        | <u>81.460</u> | <u>31.837</u> | <u>0.607</u> |

## Other Metrics

## Cite our Paper

```bash
TO BE RELEASED
```





