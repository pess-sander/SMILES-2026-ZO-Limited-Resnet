# Solution Report

## Reproducibility Instructions

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the official validation script:

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 64 \
    --n_batches 128 \
    --output results.json
```

The run uses the full allowed sample budget:

```text
128 batches * 64 samples = 8192 samples
```

## Final Solution Description

The final solution modifies these files:

- `head_init.py`: initializes the CIFAR100 classifier head from the pretrained ImageNet ResNet18 classifier.
- `zo_optimizer.py`: replaces per-parameter finite differences with SPSA and uses better hyperparameters for SGD.

The most important component is the head initialization. Instead of starting the new classifier from random weights, I manually map each CIFAR100 class to one or more semantically related ImageNet-1K classes. For each CIFAR100 class, the corresponding ImageNet classifier rows are averaged and copied into the new `fc.weight` and `fc.bias`. This gives the model a meaningful classifier before any zero-order fine-tuning. It improves the initialized-head checkpoint from about random chance to `22.21%`.

The optimizer then fine-tunes only `fc.weight` and `fc.bias`. I use SPSA because it estimates a pseudo-gradient for all active parameters using only two loss evaluations per optimizer step. A single Rademacher perturbation direction is sampled for the whole head:

```text
g = (f(x + eps * delta) - f(x - eps * delta)) / (2 * eps) * delta
```

The final update rule is plain SGD with `lr = 8e-4` and `eps = 3e-4`. More sophisticated update rules were tested, but they performed poorly and supposedly tended to overreact to the noisy SPSA estimate.

The augmentation pipeline was left untouched. Strong augmentations increase the variance of the scalar loss difference used by SPSA, which makes the pseudo-gradient less reliable.

The best budget split found was `batch_size = 64`, `n_batches = 128`. Smaller batches gave more steps but noisier loss estimates, larger batches reduced loss noise but left too few updates.

## Experiments

### Head Initialization

In all these experiments except label mapping, the bias was initialized with zeros.

| Initialization           | Accuracy (Ckpt 2) |
|--------------------------|-------------------|
| Kaiming uniform          | 1.21%             |
| Xavier uniform gain=0.05 | 1.12%             |
| Xavier uniform gain=0.1  | 1.23%             |
| Orthogonal gain=0.05     | 0.90%             |
| All zeros                | 1.00%             |
| Normal std=0.01          | 0.89%             |
| **Label mapping**        | **22.21%**        |

### Optimizer

In all optimizer experiments, label mapping for head initialization and SPSA for gradient estimation were used. Both `fc.weight` and `fc.bias` were trained.

| Optimizer                | Accuracy (Ckpt 3) |
|--------------------------|-------------------|
| **SGD**                  | **24.81%**        |
| SGD + clip               | 22.22%            |
| SGD w/ momentum          | 6.01%             |
| SGD w/ momentum + clip   | 22.23%            |
| Adam                     | 22.86%            |
| Adam + clip              | 22.51%            |

### Budget Splitting

| Batch size | Num batches | Accuracy (Ckpt 3) |
|------------|-------------|-------------------|
| 32         | 256         | 25.24%            |
| **64**     | **128**     | **27.52%**        |
| 128        | 64          | 26.05%            |
| 256        | 32          | 25.40%            |
