# Training 

This is a guideline for training the GLM-Dialog model. The training code is largely based on [SwissArmyTransformer](https://github.com/THUDM/SwissArmyTransformer) and [GLM](https://github.com/THUDM/GLM).
```bash
cd train
```

## Content

1. [Prepare](#prepare)
2. [Training](#training)
3. [Evaluation](#evaluation)

## Prepare 

### Preprocessing
You should prepare your model checkpoint and data in this step.

#### Model-Parallel-Split
Following [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)'s approach, we employed ModelParallel technique during training to split the model into N partitions to reduce memory consumption, where N must be a power of 2.
```
cd convert_tp && bash split.sh {N}
```

#### Data
```
python preprocess.py --pad-length {PAD-LENGTH}--input-jsonl {TRAIN-DATA} --output-name {}
```
The input file format is in JSON Lines, and it will be saved as a binary format in np.memmap after processing.
```json
{"input_str":"aaaa", "output_str":"bbbb", "label":0}
```
**Note** The values of 'label' must be chosen from the range of [-1, 0, 1], representing 'unknown', 'not useful', and 'useful' respectively.

## Training

To train GLM-Dialog run
```bash
bash scripts/train.sh
```
**Note** Please set DeepSpeed and other training hyperparameters in scripts/ds_config.json and scripts/train.sh respectively.

After Training, run the following command to get the final model
```
cd convert_tp && bash merge.sh
```


### Debugging Locally
If you want a training run on a subset of datas with one local GPU (instead of using torchrun), simply set the MP_SIZE to 1, WORLD_SIZE to 1 (--include=localhost:0) at ```scripts/train.sh``` and freeze most of the parameters in ```train.py```.

This can work with >=24GB GPU memory.


## Evaluation
Evaluation can be done by following the guidelines for inference in the main [README](README.md).

Please set up your local chat service and integrate it into the testing platform we provide.