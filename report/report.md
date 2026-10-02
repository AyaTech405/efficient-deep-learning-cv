# Project Report

## Efficient Deep Learning for Computer Vision: Accuracy–Efficiency Trade-offs in Image Classification

**Status:** Experimental report based on completed experiments.

## 1\. Objective

This project studies the relationship between predictive performance and computational efficiency for two pretrained image-classification architectures on CIFAR-10.



The main objectives are to:

* compare ResNet18 and MobileNetV3-Small under the same experimental conditions
* evaluate predictive performance using multiple classification metrics
* measure model parameters, estimated model size, and inference latency
* investigate a conservative post-training quantization technique
* analyze the resulting accuracy–efficiency trade-off



The project is designed as a reproducible Bachelor-level experimental study and does not claim a novel algorithm or state-of-the-art performance.



## 2\. Research question

**How can we improve the computational efficiency of a deep learning image-classification system while maintaining acceptable predictive performance?**



The study therefore considers both predictive performance and computational efficiency rather than relying on accuracy alone.

## 3\. Experimental design

|Component|Configuration|
|-|-|
|Dataset|CIFAR-10|
|Input|224×224 RGB|
|Model 1|ResNet18, ImageNet pretrained|
|Model 2|MobileNetV3-Small, ImageNet pretrained|
|Train subset|2,000 images|
|Test subset|500 images|
|Epochs|2|
|Batch size|32|
|Optimizer|AdamW|
|Learning rate|1e-4|
|Weight decay|1e-4|
|Seed|42|
|Device|CPU|



CIFAR-10 contains 60,000 color images across 10 classes. The original 32 × 32 images are resized to 224 × 224 and normalized using ImageNet normalization parameters so that the pretrained torchvision models can be used consistently.



The experiments use reproducible subsets of 2,000 training images and 500 test images. The training and evaluation parameters can be modified through command-line arguments.

## 4\. Metrics

The experiment evaluates both predictive performance and computational efficiency



## Predictive Metrics

* Accuracy
* Macro precision
* Macro recall
* Macro F1-score



## Efficiency Metrics

* Total parameter count
* Estimated model memory size
* Inference latency



Latency is hardware-dependent and should only be compared when measurements are obtained under the same machine, runtime, batch size, and measurement conditions.



## 5\. Optimization experiment

The baseline experiment is followed by dynamic INT8 post-training quantization of supported Linear layers.



The optimization is intentionally limited in scope and does not quantize the convolutional layers of the models.



The optimized model is evaluated using the same classification and efficiency metrics as the original model where supported by the runtime.



The purpose of this experiment is to investigate the practical effect of a lightweight post-training optimization rather than to propose a new quantization method.



## 6\. Results

### Baseline comparison

|Model|Accuracy|Precision|Recall|F1|Parameters|Size (MiB)|Latency (ms, batch=1)|
|-|-:|-:|-:|-:|-:|-:|-:|
|ResNet18|85.40%|85.95%|85.35%|85.42%|11,181,642|42.69|34.78|
|MobileNetV3-Small|73.20%|77.09%|73.30%|72.43%|1,528,106|5.88|16.44|

Under the experimental configuration, ResNet18 achieved higher predictive performance across the reported classification metrics.



MobileNetV3-Small used approximately 86% fewer parameters and had an approximately 86% smaller model size. Its measured baseline CPU latency was approximately 53% lower than that of ResNet18.



These results illustrate an accuracy–efficiency trade-off between the two architectures under the selected experimental conditions.

### Optimization comparison — ResNet18

|Metric|Original|Optimized|Change|
|-|-:|-:|-|
|Accuracy|85.40%|85.20%|−0.20 pp|
|Precision|85.95%|85.78%|−0.17 pp|
|Recall|85.35%|85.15%|−0.21 pp|
|F1|85.42%|85.21%|−0.21 pp|
|Size (MiB)|42.69|42.67|−0.02 MiB|
|Latency (ms, batch=1)|38.62|38.89|+0.27 ms|



For ResNet18, dynamic quantization produced only a very small reduction in model size. The measured latency increased slightly in this run, while the predictive metrics decreased by approximately 0.2 percentage points.

### Optimization comparison — MobileNetV3-Small

|Metric|Original|Optimized|Change|
|-|-:|-:|-|
|Accuracy|73.20%|72.40%|−0.80 pp|
|Precision|77.09%|76.67%|−0.41 pp|
|Recall|73.30%|72.55%|−0.75 pp|
|F1|72.43%|71.66%|−0.78 pp|
|Size (MiB)|5.88|3.58|−2.29 MiB|
|Latency (ms, batch=1)|20.09|17.29|−2.80 ms|



For MobileNetV3-Small, dynamic quantization reduced the measured model size by approximately 39% and reduced the measured CPU latency by approximately 14%.



This was accompanied by a decrease of 0.8 percentage points in accuracy and approximately 0.78 percentage points in F1-score.



**Timing note:** latency can vary between separate executions because of CPU load and runtime conditions. Therefore, the original and optimized latency values in the optimization tables are compared within the same optimization run.

## 7\. Discussion

The baseline experiment shows different accuracy–efficiency characteristics for the two architectures.



ResNet18 achieved higher predictive performance on the selected CIFAR-10 test subset. However, this was accompanied by a substantially larger parameter count, larger model size, and higher CPU inference latency.



MobileNetV3-Small required considerably fewer parameters and less memory and achieved lower baseline CPU latency, while its predictive performance was lower under the same training configuration.



The effect of dynamic quantization was architecture-dependent.



For ResNet18, the tested quantization method had a limited effect. The model size changed only slightly, and the measured latency was marginally higher after optimization. This is consistent with the fact that the convolutional layers were not quantized and remained in floating point.



For MobileNetV3-Small, the same optimization produced a more visible reduction in model size and measured latency. However, this reduction was accompanied by a small decrease in predictive performance.



These results show that computational efficiency should not be evaluated using a single metric. Reducing model size or latency can involve a change in predictive performance, and the practical effect of an optimization depends on the architecture and the layers affected by the optimization.



The findings are specific to the CIFAR-10 subset, training configuration, software environment, and CPU used for these experiments.

## 8\. Limitations

The experiment uses a relatively small subset of CIFAR-10 and a limited training budget. Therefore, the results should be interpreted as a lightweight experimental study rather than a definitive benchmark.



Additional limitations include:

* CIFAR-10 may not represent real-world computer-vision applications.
* Resizing 32 × 32 images to 224 × 224 introduces an experimental preprocessing choice.
* Inference latency is strongly dependent on hardware and runtime configuration.
* Dynamic quantization in this project affects only supported Linear layers and does not quantize the convolutional backbone.
* A single training run does not fully capture variability across random seeds.
* Results may vary across different hardware, operating systems, and software versions.
* The two models were trained for only two epochs, so the measured predictive performance should not be interpreted as their fully optimized performance.



Repeated experiments with multiple seeds, longer training, larger datasets, and additional optimization methods would provide stronger evidence.







## 9\. Conclusion

This study compared ResNet18 and MobileNetV3-Small in terms of predictive performance and computational efficiency.



Under the selected experimental configuration, ResNet18 achieved higher classification performance, while MobileNetV3-Small required substantially fewer parameters, had a smaller model size, and achieved lower baseline CPU inference latency.



The dynamic INT8 post-training quantization experiment produced different effects for the two architectures. Its effect was limited for ResNet18, while MobileNetV3-Small showed a substantial reduction in model size and measured latency, accompanied by a modest decrease in predictive performance.



These results demonstrate that improving computational efficiency involves a trade-off between predictive performance and resource requirements. The appropriate choice therefore depends on the requirements of the target deployment environment rather than on a single evaluation metric.



The project provides a reproducible Bachelor-level experimental study of accuracy–efficiency trade-offs in pretrained computer-vision models.

