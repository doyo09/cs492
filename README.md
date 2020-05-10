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