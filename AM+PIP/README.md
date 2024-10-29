<h1 align="center"> Learning to Handle Complex Constraints for Vehicle Routing Problems </h1>

<p align="center">
<a href="https://neurips.cc/Conferences/2024"><img alt="License" src="https://img.shields.io/static/v1?label=NeurIPS'24&message=Vancouver&color=purple&style=flat-square"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://neurips.cc/virtual/2024/poster/95638"><img alt="License" src="https://img.shields.io/static/v1?label=NeurIPS&message=Poster&color=blue&style=flat-square"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://arxiv.org/abs/2410.21066"><img src="https://img.shields.io/static/v1?label=ArXiv&message=PDF&color=red&style=flat-square" alt="Paper"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href=""><img alt="License" src="https://img.shields.io/static/v1?label=Download&message=Slides&color=orange&style=flat-square"></a>&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://github.com/jieyibi/PIP-constraint/blob/main/LICENSE"><img 
alt="License" src="https://img.shields.io/static/v1?label=License&message=MIT&color=rose&style=flat-square"></a>
</p>

---

This is the PyTorch code for the **Proactive Infeasibility Prevention (PIP)** 
framework implemented on [AM](https://github.com/wouterkool/attention-learn-to-route).


## Train

```shell
# Default: --graph_size=50 --hardness=hard --CUDA_VISIBLE_ID=0

# 1. AM*
python run.py --graph_size={PROBLEM_SIZE} --hardness={HARDNESS}

# 2. AM* + PIP
python run.py --graph_size={PROBLEM_SIZE} --hardness={HARDNESS} --generate_PI_mask

# 3. AM* + PIP-D
python run.py --graph_size={PROBLEM_SIZE} --hardness={HARDNESS} --generate_PI_mask --pip_decoder

# Note: If you want to resume, please add arguments: --pip_checkpoint and --resume
```

## Evaluation

For evaluation, please download the data or generate datasets first. 
Pretrained models are provided in the folder `./pretrained/`.

```shell
# Default: --graph_size=50 --hardness=hard --CUDA_VISIBLE_ID=0

# 1. AM*

# If you want to evaluate on your own dataset,
python eval.py --datasets={DATASET} --model={MODEL_PATH}
# Optional: add `--val_solution_path` to calculate optimality gap.

# If you want to evaluate on the provided dataset,
python eval.py --graph_size={PROBLEM_SIZE} --hardness={HARDNESS} --model={MODEL_PATH}

# 2. AM* + PIP(-D)

# If you want to evaluate on your own dataset,
python eval.py --datasets={DATASET} --model={MODEL_PATH} --generate_PI_mask
# Optional: add `--val_solution_path` to calculate optimality gap.

# If you want to evaluate on the provided dataset,
python eval.py --graph_size={PROBLEM_SIZE} --hardness={HARDNESS} --model={MODEL_PATH} --generate_PI_mask


# Please set your own `--eval_batch_size` based on your GPU memory constraint.
```


 
