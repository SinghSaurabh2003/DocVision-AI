# 📄 DocVision AI
### Intelligent Document Classification using Transfer Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Torchvision](https://img.shields.io/badge/Torchvision-TransferLearning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

## 📌 Overview

**DocVision AI** is a deep learning-based document classification system capable of automatically categorizing scanned document images into predefined document categories using **Transfer Learning** with **ResNet18**.

The project provides an end-to-end machine learning pipeline including:

- Dataset preprocessing
- Model training
- Model evaluation
- Checkpoint management
- Real-time prediction
- Interactive Streamlit deployment

---

## 🚀 Features

- Transfer Learning using ImageNet pretrained ResNet18
- Automatic classification of scanned documents
- Complete training pipeline
- Model checkpointing
- Performance evaluation
- Confusion Matrix generation
- Classification Report
- Single image prediction
- Top-K prediction probabilities
- Streamlit Web Application
- GPU supported training and inference

---

# 📂 Dataset

Dataset Used:

**RVL-CDIP Document Image Dataset**

Classes:

- Advertisement
- Budget
- Email
- File Folder
- Form
- Handwritten
- Invoice
- Letter
- Memo

---

# 🧠 Model Architecture

- Backbone : ResNet18
- Pretrained : ImageNet
- Transfer Learning
- Frozen Layers : Layer1–Layer3
- Fine-tuned : Layer4 + Fully Connected Layer

---

# ⚙️ Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- Transfer Learning
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Pillow
- Matplotlib
- Git
- GitHub

---

# 📁 Project Structure

```text
DocVision-AI/
│
├── app.py
├── predict.py
├── evaluate.py
├── train.py
├── requirements.txt
│
├── checkpoints/
│     ├── best_model.pth
│     └── last_model.pth
│
├── src/
│     ├── config.py
│     ├── model.py
│     ├── dataset.py
│     ├── dataloader.py
│     ├── trainer.py
│     └── utils.py
│
├── results/
│
└── README.md
```

---

# 🏗️ Development Journey

## Version 1

### Objective

Develop an initial transfer learning model for document classification.

### Pipeline

- Dataset preprocessing
- Transfer Learning using ResNet18
- Layer4 fine-tuning
- Evaluation pipeline
- Streamlit deployment

### Limitation

During evaluation, the model produced:

- Validation Accuracy : **91.83%**
- Test Accuracy : **81.39%**

The Memo class achieved **0% accuracy**.

Investigation revealed that the training dataset contained only **6 Memo images**, while the test dataset contained approximately **2,500 Memo images**, resulting in severe class imbalance.

---

## Version 2

### Improvements

The complete training dataset was verified and corrected.

The missing Memo class images were restored, resulting in a balanced dataset containing approximately **20,000 images per class**.

The model was resumed from the best checkpoint and fine-tuned further without retraining from scratch.

### Final Results

| Metric | Score |
|---------|-------|
| Accuracy | **89.66%** |
| Precision | **89.68%** |
| Recall | **89.66%** |
| F1 Score | **89.65%** |

Memo class accuracy improved from:

```
0%
```

to

```
83.87%
```

---

# 📊 Final Performance

| Class | Accuracy |
|---------|----------|
| Advertisement | 93.24% |
| Budget | 88.86% |
| Email | 97.22% |
| File Folder | 97.35% |
| Form | 80.57% |
| Handwritten | 94.63% |
| Invoice | 85.75% |
| Letter | 85.19% |
| Memo | 83.87% |

Overall Test Accuracy:

**89.66%**

---

# 🌐 Streamlit Application

The application allows users to:

- Upload document images
- Predict document category
- Display confidence score
- View Top-3 predictions
- Perform real-time inference

---

# 📈 Evaluation

The evaluation pipeline automatically generates:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix
- Per-class Accuracy
- Prediction CSV

---

# ⚠️ Current Limitations

- Supports only nine document categories.
- Relies solely on visual document appearance; textual content is not analyzed.
- Performance may decrease on heavily distorted, low-resolution, or handwritten documents outside the training distribution.
- Single-label classification only.
- ResNet18 provides a good balance between speed and accuracy but is not the most powerful backbone available.

---

# 🔮 Future Scope

Future improvements include:

- Support for all 16 RVL-CDIP document categories.
- Adoption of stronger vision backbones such as EfficientNet, ConvNeXt, or Vision Transformers (ViT).
- Integration of OCR for combining visual and textual information.
- Multi-page document classification.
- Explainable AI using Grad-CAM to visualize model attention.
- Active learning to continuously improve performance with user feedback.
- REST API deployment using FastAPI.
- Docker containerization for scalable deployment.
- Cloud deployment with automated CI/CD.
- Support for multilingual document classification.

---

# 📚 Learning Outcomes

Through this project, the following concepts were explored:

- Transfer Learning
- Deep Learning
- Image Classification
- PyTorch
- Torchvision
- Model Checkpointing
- Dataset Engineering
- Handling Class Imbalance
- Evaluation Metrics
- Streamlit Deployment
- Git & Git LFS

---

# 👨‍💻 Author

**Saurabh Singh Chhonkar**

M.Tech Computer Science & Engineering

Motilal Nehru National Institute of Technology, Allahabad

---

## ⭐ If you found this project useful, consider giving it a Star!
