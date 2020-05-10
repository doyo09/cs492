# CS492(H) DL4RW CV Project
Semi-supervised learning on naver fashion eval task.

File an issue or contact me if you cannot reproduce a result.

## Example
```
nsml run -d fashion_eval -e train.py -a "<ARGS>"
```

**fixmatch**
```
nsml run -d fashion_eval -e train.py -a "--fixmatch --batchsize=64 --N=2 --M=10 --ema --sgd"
```


**hardmixmatch**

with resnet50
```
nsml run -d fashion_eval -e train.py -a "--hardmixmatch --hardmixmatch_threshold .8 --lr .03 --batchsize=64 --N=2 --M=10 --ema --sgd"
```

with resnet18
```
nsml run -d fashion_eval -e train.py -a "--model_res18 --hardmixmatch --hardmixmatch_threshold .8 --lr .03 --batchsize=64 --N=2 --M=10 --ema --sgd"
```
