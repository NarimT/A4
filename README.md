# 🔍 BERT & Sentence-BERT: Text Similarity and NLI

This project implements **BERT from scratch** and fine-tunes it as **Sentence-BERT (S-BERT)** for Natural Language Inference (NLI) and semantic text similarity. It includes a web application for interactive NLI prediction.

> **Course:** A4: Do you AGREE?  
> **Author:** NarimT

---

## 📁 Project Structure

```
.
├── Bert (1).ipynb                # Task 1: BERT pre-training from scratch
├── Task2-Task3.ipynb             # Task 2: S-BERT fine-tuning & Task 3: Evaluation
├── bert_model_with_config.pth    # Saved BERT model weights + config
├── classifier_head.pth           # Saved S-BERT classifier head weights
├── word2id.json                  # Vocabulary mapping (word → token ID)
├── README.md                     # This file
└── app/                          # Task 4: Web application
    ├── app.py                    # Flask backend
    ├── requirements.txt          # Python dependencies
    └── templates/
        └── index.html            # Frontend UI
```

---

## 📋 Task Overview

### Task 1: Training BERT from Scratch (2 points)
- Implemented BERT (Bidirectional Encoder Representations from Transformers) from scratch using PyTorch
- Trained on **100,000 samples** from the [BookCorpus](https://huggingface.co/datasets/bookcorpus) dataset
- Pre-training objectives: **Masked Language Model (MLM)** + **Next Sentence Prediction (NSP)**
- Model saved as `bert_model_with_config.pth`

### Task 2: Sentence-BERT with Siamese Network (3 points)
- Loaded pre-trained BERT weights from Task 1
- Implemented S-BERT with **Classification Objective Function (SoftmaxLoss)**:

$$o = \text{softmax}(W^T \cdot (u, v, |u - v|))$$

- Fine-tuned on [SNLI](https://huggingface.co/datasets/snli) + [MNLI](https://huggingface.co/datasets/glue/viewer/mnli) datasets
- Predicts: **Entailment**, **Neutral**, **Contradiction**

### Task 3: Evaluation and Analysis (1 point)
- Classification report and confusion matrix on test/validation sets
- Discussion of limitations, challenges, and proposed improvements

### Task 4: Web Application (1 point)
- Flask-based web app with two input boxes (premise & hypothesis)
- Predicts NLI label with confidence scores and cosine similarity

---

## 🔧 Model Configuration

| Parameter | Value |
|-----------|-------|
| `n_layers` | 6 |
| `n_heads` | 4 |
| `d_model` | 512 |
| `d_ff` | 2048 |
| `d_k` | 128 |
| `max_len` | 128 |
| `vocab_size` | 23,068 |

---

## 📊 Datasets

| Dataset | Source | Usage |
|---------|--------|-------|
| BookCorpus | [HuggingFace](https://huggingface.co/datasets/bookcorpus) | BERT pre-training (100k samples) |
| SNLI | [HuggingFace](https://huggingface.co/datasets/snli) | S-BERT fine-tuning |
| MNLI | [HuggingFace](https://huggingface.co/datasets/glue/viewer/mnli) | S-BERT fine-tuning |

**Credits:**
- BookCorpus: Zhu et al., "Aligning Books and Movies", arXiv 2015
- SNLI: Bowman et al., "A large annotated corpus for learning natural language inference", EMNLP 2015
- MNLI: Williams et al., "A Broad-Coverage Challenge Corpus for Sentence Understanding", NAACL 2018

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch numpy flask scikit-learn datasets transformers tqdm matplotlib seaborn
```

### Running the Notebooks

1. **Task 1** — Open and run `Bert (1).ipynb` to pre-train BERT from scratch
2. **Task 2 & 3** — Open and run `Task2-Task3.ipynb` for S-BERT training and evaluation

### Running the Web Application

```bash
cd app
pip install -r requirements.txt
python app.py
```

Then open your browser at: **http://localhost:5001**

> **Note:** If port 5001 is also in use, change the port number in the last line of `app/app.py`.

---

## 🌐 Web App Usage

1. Enter a **Premise** sentence in the first text box
2. Enter a **Hypothesis** sentence in the second text box
3. Click **"Predict NLI Label"**
4. View the predicted label, confidence scores, and cosine similarity

**Example:**
- **Premise:** A man is playing a guitar on stage.
- **Hypothesis:** The man is performing music.
- **Predicted Label:** Entailment

---

## 📈 Results

| Metric | Test Set | Validation Set |
|--------|----------|----------------|
| Accuracy | 40.91% | 33.40% |
| Macro F1 | 0.1935 | 0.1669 |

> **Note:** Performance is limited due to small pre-training data (100k samples), word-level tokenization (~23k vocab), and small fine-tuning dataset (1000 samples). See Task 3 discussion in the notebook for detailed analysis and proposed improvements.

---

## 📚 References

1. Devlin et al., ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://aclanthology.org/N19-1423.pdf), NAACL 2019
2. Reimers & Gurevych, ["Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"](https://aclanthology.org/D19-1410/), EMNLP 2019
3. [Pinecone S-BERT Tutorial](https://www.pinecone.io/learn/series/nlp/train-sentence-transformers-softmax/) (Reference Code)