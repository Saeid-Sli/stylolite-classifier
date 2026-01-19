# ResNet & ResNeXt Image Classification Project

This repository contains three Python scripts for training, evaluating, and fine-tuning deep learning models based on **ResNet-50** and **ResNeXt-50** architectures using TensorFlow/Keras. The goal of the project is to classify rock images into five different geological classes.

---

## Repository Structure

```
├── pre_ResNet-50.py
├── pre_ResNeXt50.py
├── retrain_resnet_resnext.py
├── README.md
└── LICENSE
```

---

## File Descriptions

### 1. `pre_ResNet-50.py`

This script performs **initial training (transfer learning)** using a pre-trained **ResNet-50** model.

**Key features:**

* Uses ImageNet pre-trained weights
* Image data augmentation (shear, zoom, horizontal flip)
* Custom fully connected layers
* SGD optimizer
* Accuracy and loss visualization
* Confusion matrix computation

**Classification classes:**

* intact Rock
* stylolite
* horizontal plug
* vertical plug
* Crack

---

### 2. `pre_ResNeXt50.py`

This script is similar to the ResNet version but uses **ResNeXt-50** as the backbone network.

**Main differences from ResNet-50:**

* Higher cardinality architecture
* Improved feature representation capability

Other steps include:

* Data augmentation
* Training with SGD optimizer
* Model evaluation and confusion matrix visualization

---

### 3. `retrain_resnet_resnext.py`

This script is designed for **fine-tuning** previously trained ResNet or ResNeXt models.

**Capabilities:**

* Dynamic learning rate scheduling (`LearningRateScheduler`)
* SGD with momentum
* Dropout layers to reduce overfitting
* Multiple fine-tuning strategies (commented and optional)

**Fine-tuning options included:**

* Freezing selected base model layers
* Switching optimizers (e.g., Adam)
* Advanced data augmentation
* Adding additional dense layers

---

## Requirements

* Python 3.8+
* TensorFlow 2.x
* NumPy
* Matplotlib
* Seaborn
* scikit-learn
* Google Colab (recommended)

Install dependencies:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

---

## Dataset Structure

The dataset must be organized as follows (Google Drive):

```
DataSplit/
├── train/
│   ├── intact Rock/
│   ├── stylolite/
│   ├── horizontal plug/
│   ├── vertical plug/
│   └── Crack/
└── validation/
    ├── intact Rock/
    ├── stylolite/
    ├── horizontal plug/
    ├── vertical plug/
    └── Crack/
```

---

## How to Run

1. Open the desired script in Google Colab
2. Mount Google Drive
3. Verify dataset paths
4. Run the script

---

## Outputs

* Training and validation accuracy plots
* Training and validation loss plots
* Final validation accuracy
* Confusion matrix for model performance analysis

---

## 📜 License

This project is released under the **MIT License**.  
You may freely use and adapt the code for research and educational purposes.

See the [LICENSE](./LICENSE) file for full details.
