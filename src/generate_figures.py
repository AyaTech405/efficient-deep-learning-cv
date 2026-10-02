import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# Output directory
# ============================================================

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Experimental results
# ============================================================

models = ["ResNet18", "MobileNetV3-Small"]

# Baseline results
accuracy = [0.854, 0.732]
f1 = [0.8541515803, 0.7243377483]
parameters = [11_181_642, 1_528_106]
model_size = [42.69135, 5.87572]
latency = [34.7762, 16.43569]

# Training results
epochs = [1, 2]

resnet_loss = [1.1446, 0.3868]
resnet_accuracy = [0.7860, 0.8540]
resnet_f1 = [0.7836, 0.8542]

mobilenet_loss = [1.8152, 0.8804]
mobilenet_accuracy = [0.6440, 0.7320]
mobilenet_f1 = [0.6240, 0.7243]

# Optimization experiment
quant_models = ["ResNet18", "MobileNetV3-Small"]

quant_original_accuracy = [0.854, 0.732]
quantized_accuracy = [0.852, 0.724]

quant_original_size = [42.69135, 5.87572]
quantized_size = [42.67178, 3.58272]

quant_original_latency = [38.61708, 20.08688]
quantized_latency = [38.89054, 17.28672]


# ============================================================
# Helper
# ============================================================

def save_figure(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 1. Accuracy comparison
# ============================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    [x * 100 for x in accuracy]
)

plt.ylabel("Accuracy (%)")
plt.title("Accuracy Comparison")

for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value * 100:.1f}%",
        ha="center",
        va="bottom"
    )

plt.grid(axis="y", alpha=0.25)

save_figure("accuracy_comparison.png")


# ============================================================
# 2. Macro F1 comparison
# ============================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    [x * 100 for x in f1]
)

plt.ylabel("Macro F1 (%)")
plt.title("Macro F1 Comparison")

for bar, value in zip(bars, f1):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value * 100:.1f}%",
        ha="center",
        va="bottom"
    )

plt.grid(axis="y", alpha=0.25)

save_figure("f1_comparison.png")


# ============================================================
# 3. Parameter count
# ============================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    [x / 1_000_000 for x in parameters]
)

plt.ylabel("Parameters (millions)")
plt.title("Parameter Count Comparison")

for bar, value in zip(bars, parameters):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value / 1_000_000:.2f}M",
        ha="center",
        va="bottom"
    )

plt.grid(axis="y", alpha=0.25)

save_figure("parameter_comparison.png")


# ============================================================
# 4. Model size
# ============================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    model_size
)

plt.ylabel("Estimated model size (MiB)")
plt.title("Model Size Comparison")

for bar, value in zip(bars, model_size):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.grid(axis="y", alpha=0.25)

save_figure("model_size_comparison.png")


# ============================================================
# 5. Baseline inference latency
# ============================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(
    models,
    latency
)

plt.ylabel("CPU inference latency (ms)")
plt.title("Baseline CPU Inference Latency")

for bar, value in zip(bars, latency):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.grid(axis="y", alpha=0.25)

save_figure("latency_comparison.png")


# ============================================================
# 6. Accuracy vs latency
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    latency,
    [x * 100 for x in accuracy],
    s=100
)

for x, y, label in zip(
    latency,
    [x * 100 for x in accuracy],
    models
):
    plt.annotate(
        label,
        (x, y),
        xytext=(7, 7),
        textcoords="offset points"
    )

plt.xlabel("CPU inference latency (ms)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Inference Latency")

plt.grid(alpha=0.25)

save_figure("accuracy_vs_latency.png")


# ============================================================
# 7. Quantization accuracy
# ============================================================

x = np.arange(len(quant_models))
width = 0.35

plt.figure(figsize=(8, 5))

bars1 = plt.bar(
    x - width / 2,
    [x * 100 for x in quant_original_accuracy],
    width,
    label="Original"
)

bars2 = plt.bar(
    x + width / 2,
    [x * 100 for x in quantized_accuracy],
    width,
    label="Quantized"
)

plt.xticks(x, quant_models)
plt.ylabel("Accuracy (%)")
plt.title("Quantization: Accuracy")
plt.legend()

plt.grid(axis="y", alpha=0.25)

save_figure("quantization_accuracy.png")


# ============================================================
# 8. Quantization model size
# ============================================================

plt.figure(figsize=(8, 5))

bars1 = plt.bar(
    x - width / 2,
    quant_original_size,
    width,
    label="Original"
)

bars2 = plt.bar(
    x + width / 2,
    quantized_size,
    width,
    label="Quantized"
)

plt.xticks(x, quant_models)
plt.ylabel("Model size (MiB)")
plt.title("Quantization: Model Size")
plt.legend()

plt.grid(axis="y", alpha=0.25)

save_figure("quantization_model_size.png")


# ============================================================
# 9. Quantization latency
# ============================================================

plt.figure(figsize=(8, 5))

bars1 = plt.bar(
    x - width / 2,
    quant_original_latency,
    width,
    label="Original"
)

bars2 = plt.bar(
    x + width / 2,
    quantized_latency,
    width,
    label="Quantized"
)

plt.xticks(x, quant_models)
plt.ylabel("CPU inference latency (ms)")
plt.title("Quantization: Inference Latency")
plt.legend()

plt.grid(axis="y", alpha=0.25)

save_figure("quantization_latency.png")


# ============================================================
# 10. Training loss
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    resnet_loss,
    marker="o",
    label="ResNet18"
)

plt.plot(
    epochs,
    mobilenet_loss,
    marker="o",
    label="MobileNetV3-Small"
)

plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("Training Loss")

plt.xticks(epochs)
plt.legend()
plt.grid(alpha=0.25)

save_figure("training_loss.png")


# ============================================================
# 11. Training accuracy
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    [x * 100 for x in resnet_accuracy],
    marker="o",
    label="ResNet18"
)

plt.plot(
    epochs,
    [x * 100 for x in mobilenet_accuracy],
    marker="o",
    label="MobileNetV3-Small"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy")

plt.xticks(epochs)
plt.legend()
plt.grid(alpha=0.25)

save_figure("training_accuracy.png")


# ============================================================
# 12. Training Macro F1
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    [x * 100 for x in resnet_f1],
    marker="o",
    label="ResNet18"
)

plt.plot(
    epochs,
    [x * 100 for x in mobilenet_f1],
    marker="o",
    label="MobileNetV3-Small"
)

plt.xlabel("Epoch")
plt.ylabel("Macro F1 (%)")
plt.title("Training Macro F1")

plt.xticks(epochs)
plt.legend()
plt.grid(alpha=0.25)

save_figure("training_f1.png")


print()
print("Figures generated successfully!")
print()
print("Location:")
print(OUTPUT_DIR.resolve())
print()
print("Generated files:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(" -", file.name)