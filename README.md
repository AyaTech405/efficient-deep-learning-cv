# Efficient Deep Learning for Computer Vision

## Accuracy–Efficiency Trade-offs in Image Classification

A Bachelor-level experimental research project studying the trade-off between predictive performance and computational efficiency in pretrained deep learning models for image classification.

## Motivation

Deep learning models can achieve strong predictive performance while requiring substantial compute and memory. For practical AI systems, model selection is therefore not only about accuracy: model size and inference latency also matter.

This project compares a conventional residual network with a lightweight mobile-oriented architecture under a controlled CIFAR-10 experiment.

## Research Question

> How can we improve the computational efficiency of a deep learning image-classification system while maintaining acceptable predictive performance?

This project is an experimental benchmark. It does not claim a new algorithm or state-of-the-art performance.

## Models

* **ResNet18** pretrained on ImageNet
* **MobileNetV3-Small** pretrained on ImageNet

The final classification heads are adapted to the 10 CIFAR-10 classes.

## Dataset

CIFAR-10 contains 60,000 color images across 10 classes.

The implementation:

* resizes images to `224 × 224`
* applies ImageNet normalization
* uses pretrained torchvision backbones
* supports reproducible dataset subsets

The default experimental configuration uses:

* 2,000 training images
* 500 test images
* batch size of 32
* fixed random seed of 42
* 2 training epochs by default

The subset sizes, batch size, number of epochs, and other training parameters can be changed from the command line.

## Methodology

The experimental workflow consists of the following steps:

1. Load CIFAR-10 with reproducible subsets.
2. Resize the images to the input resolution expected by the pretrained models.
3. Load ImageNet-pretrained ResNet18 and MobileNetV3-Small.
4. Adapt the final classification layers to the 10 CIFAR-10 classes.
5. Fine-tune each model using the same experimental configuration.
6. Evaluate classification performance on the test subset.
7. Measure model parameters and estimated model size.
8. Benchmark inference latency under controlled conditions.
9. Compare the two architectures using structured metrics and visualizations.
10. Apply post-training dynamic INT8 quantization to supported `Linear` layers.
11. Compare the original and optimized model where the quantization runtime is supported.

The same dataset split, training configuration, and evaluation procedure should be used for both models when making a direct comparison.

## Metrics

The project evaluates both predictive performance and computational efficiency.

### Predictive metrics

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score

### Efficiency metrics

* Trainable parameter count
* Estimated model size in MiB
* Inference latency in milliseconds

Latency is hardware-dependent. Meaningful comparisons therefore require the same machine, runtime, batch size, preprocessing configuration, and measurement protocol.

## Post-Training Quantization

The project includes a conservative optimization experiment using dynamic INT8 quantization.

The quantization experiment targets supported `Linear` layers after training.

It deliberately does not introduce:

* ONNX
* TensorRT
* pruning
* knowledge distillation
* distributed training
* custom quantization algorithms

Because most of the computation in these CNN architectures occurs in convolutional layers, the improvement obtained from quantizing only supported `Linear` layers may be modest.

The purpose of this experiment is therefore to study the practical effect of a simple post-training optimization rather than to claim a complete model-compression solution.

## Project Structure

```text
efficient-deep-learning-cv/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── optimize.py
│
├── results/
│   └── figures/
│       └── .gitkeep
│
├── report/
│   └── report.md
│
└── paper/
    └── research_paper.md
```

### Main modules

* `src/data.py` — dataset loading, preprocessing, reproducible subsets, and random seed configuration.
* `src/models.py` — model construction and model-size utilities.
* `src/train.py` — model fine-tuning and checkpoint generation.
* `src/evaluate.py` — checkpoint evaluation and classification metrics.
* `src/benchmark.py` — inference benchmarking and model comparison.
* `src/optimize.py` — post-training dynamic quantization experiment.

Generated checkpoints, downloaded datasets, virtual environments, and temporary Python files are excluded from version control through `.gitignore`.

## Installation

From the project root, create a virtual environment:

```bash
python -m venv .venv
```

### Windows PowerShell

Activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution for the current session, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate the environment again:

```powershell
.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduction

All commands below should be executed from the project root.

### 1. Train ResNet18

```powershell
python -m src.train --model resnet18
```

### 2. Train MobileNetV3-Small

```powershell
python -m src.train --model mobilenetv3-small
```

The first run downloads:

* the CIFAR-10 dataset
* the pretrained ImageNet weights required by the models

These files are intentionally excluded from Git.

Training produces checkpoints under:

```text
results/checkpoints/
```

and training history under:

```text
results/training_history.csv
```

These generated artifacts are also excluded from version control.

### 3. Evaluate a ResNet18 checkpoint

After training:

```powershell
python -m src.evaluate --model resnet18 --checkpoint results/checkpoints/resnet18_best.pt
```

### 4. Evaluate a MobileNetV3-Small checkpoint

```powershell
python -m src.evaluate --model mobilenetv3-small --checkpoint results/checkpoints/mobilenetv3_small_best.pt
```

### 5. Benchmark both models

After both checkpoints have been generated:

```powershell
python -m src.benchmark --resnet-checkpoint results/checkpoints/resnet18_best.pt --mobilenet-checkpoint results/checkpoints/mobilenetv3_small_best.pt
```

On Windows PowerShell, the command can also be entered as a single line, as shown above.

### 6. Run the optimization experiment

For ResNet18:

```powershell
python -m src.optimize --model resnet18 --checkpoint results/checkpoints/resnet18_best.pt
```

The optimization script applies dynamic INT8 quantization to supported `Linear` layers and compares the optimized model with the original model where supported by the runtime.

## Experimental Scale

The default configuration is intentionally lightweight so that the project can run on a normal laptop.

The default training configuration uses:

```text
Training subset: 2,000 images
Test subset:       500 images
Batch size:         32
Epochs:              2
Learning rate:   1e-4
Weight decay:    1e-4
Random seed:        42
```

For a stronger experiment, the subset size and number of epochs can be increased.

For example:

```powershell
python -m src.train --model resnet18 --train-size 10000 --test-size 2000 --epochs 5
```

The equivalent configuration should then be used for MobileNetV3-Small:

```powershell
python -m src.train --model mobilenetv3-small --train-size 10000 --test-size 2000 --epochs 5
```

When comparing the two architectures, the experimental settings should remain identical whenever possible.

## Reproducibility

The project uses a fixed random seed to make dataset subset selection and training-related randomness more reproducible.

The default seed is:

```text
42
```

However, exact numerical results can still vary depending on:

* CPU or GPU hardware
* PyTorch version
* torchvision version
* operating system
* numerical libraries
* runtime configuration
* number of threads
* training duration

Therefore, reproducibility should be understood as controlled experimental reproducibility rather than a guarantee of bit-identical results on every machine.

## Results

Final experimental results are intentionally not included until the experiments have actually been executed.

No experimental values are fabricated or inserted in advance.

After the final experiments are completed, the analysis should report the measured values for both architectures.

## Expected Baseline Comparison

| Model | Accuracy | Precision | Recall | F1 | Parameters | Size (MiB) | Latency (ms) |
|-------|----------|-----------|--------|----|------------|------------|--------------|
| ResNet18 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MobileNetV3-Small | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Expected optimization comparison

| Metric       | Original | Quantized |
| ------------ | -------: | --------: |
| Accuracy     |      TBD |       TBD |
| Precision    |      TBD |       TBD |
| Recall       |      TBD |       TBD |
| F1           |      TBD |       TBD |
| Size (MiB)   |      TBD |       TBD |
| Latency (ms) |      TBD |       TBD |

These tables should only be populated after the corresponding experiments have been executed.

## Planned Visualizations

The final analysis is expected to include several visual comparisons:

* Accuracy comparison
* Macro F1 comparison
* Parameter-count comparison
* Model-size comparison
* Inference-latency comparison
* Accuracy versus latency
* Accuracy versus model size
* Original versus quantized model comparison

The figures will be stored under:

```text
results/figures/
```

## Discussion

The central analysis does not focus only on which model obtains the highest classification accuracy.

Instead, the project examines the relationship between:

* predictive performance
* model size
* parameter count
* inference latency
* optimization effects

A model with slightly different predictive performance may have a substantially different computational footprint. The purpose of the experiment is therefore to analyze the measured accuracy–efficiency trade-off rather than reduce the comparison to a single metric.

The final discussion should be based only on the actual experimental measurements obtained under the selected configuration and hardware.

## Limitations

This project has several limitations.

### Dataset limitations

CIFAR-10 is a relatively small benchmark and does not represent every real-world computer-vision task.

The original CIFAR-10 images are only `32 × 32` pixels. Resizing them to `224 × 224` is an experimental preprocessing choice required to use the selected ImageNet-pretrained architectures consistently.

### Training limitations

The default configuration uses a relatively small subset and a limited number of training epochs.

This makes the experiment feasible on modest hardware, but it also means that the default experiment should be interpreted as a lightweight research demonstration rather than a definitive benchmark.

### Hardware limitations

Inference latency depends strongly on:

* CPU/GPU hardware
* PyTorch version
* runtime configuration
* batch size
* number of threads
* operating system

Latency values should therefore not be generalized beyond the environment in which they were measured.

### Quantization limitations

The dynamic quantization experiment targets supported `Linear` layers.

The convolutional backbone remains in floating-point precision.

Consequently, the optimization does not represent full CNN quantization and may provide only limited efficiency improvements.

### Statistical limitations

A single training run does not fully characterize the variability of the models.

Repeated experiments with multiple random seeds would provide stronger evidence and allow confidence intervals or other measures of variability to be reported.

## Future Work

Possible extensions include:

* larger CIFAR-10 training and test subsets
* longer training schedules
* repeated experiments with multiple random seeds
* confidence intervals
* controlled CPU and GPU comparisons
* more rigorous latency measurement protocols
* evaluation on an additional computer-vision dataset
* comparison with additional lightweight architectures
* investigation of convolution-aware quantization
* pruning experiments
* knowledge distillation
* additional model-compression techniques

These extensions are intentionally outside the scope of the minimal Bachelor-level project.

## Academic Positioning

This project is an **experimental Bachelor-level research project**.

It should be presented honestly as a reproducible comparison and analysis rather than as:

* a novel deep learning algorithm
* a state-of-the-art contribution
* a new quantization method
* a production-ready deployment system

The main academic objective is to understand and experimentally analyze the relationship between predictive performance and computational efficiency in deep learning models.

The project emphasizes:

* reproducibility
* controlled experimentation
* meaningful evaluation metrics
* computational efficiency
* honest interpretation of results

## Research Scope

The project lies at the intersection of:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Computer Vision
* Efficient AI
* Model Optimization
* AI Systems and Infrastructure

The experimental design is intentionally small enough to be completed at Bachelor level while introducing concepts relevant to research in efficient machine learning and resource-aware AI systems.

## Conclusion

The project investigates a practical question in modern deep learning: how predictive performance and computational efficiency interact when selecting an image-classification model.

By comparing ResNet18 and MobileNetV3-Small under a controlled CIFAR-10 experiment, the project provides a reproducible framework for studying this trade-off.

The final conclusion will be written only after the complete experiments are executed and should directly answer the research question using measured evidence.

No unsupported performance claims will be made.
