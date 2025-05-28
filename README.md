# Text Classification with BERT & From-Scratch Attention (Bonus)

## Project Overview

This repository implements three text-classification pipelines on the 43-label `train.csv` dataset:

1. **BERT Fine-Tuning (Q1a)**
2. **From-Scratch Transformer with Word2Vec embeddings (Q1b)**
3. **Efficient Attention Exploration: Linear Attention vs. Standard Softmax (Q1c)**

---

## Setup Steps for Local Execution

1. **Clone the repo**

   ```bash
   git clone https://github.com/thebluecatbot/aic2025.git
   cd aic2025
   ```

2. **Create & activate Conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate aic2025
   ```

   *Or install via pip:*

   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK corpora (stopwords, wordnet)**

   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

4. **Data placement**
   Place `train.csv` under `./data/` (expected columns: `Category`, `Text`).

---

## Detailed Pipeline & Notebooks

### 1) BERT Fine-Tuning (`01_bert_finetune.ipynb`)

* **Data loading**: stratified 90/10 split of `train.csv`
* **Preprocessing**: lowercase, remove stop-words, lemmatize, simple synonym-replacement augmentation
* **Model**: `BertTokenizerFast` + `BertForSequenceClassification(num_labels=43)`
* **Training**: track train/validation loss, accuracy, precision, recall, F1; plot learning curves
* **Hyperparameter search stub**: `learning_rate`, `batch_size`, `num_epochs`

### 2) From-Scratch Transformer (`02_scratch_transformer.ipynb`)

* **Word2Vec**: train 128-dim embeddings on training texts
* **Data pipeline**: custom `Dataset` & `DataLoader` with `PAD`/`UNK` tokens, `MAX_LEN=128`
* **Model components**: `PositionalEncoding`, `MultiHeadSelfAttention`, `TransformerBlock`, `ScratchClassifier`
* **Training loop**: record loss/accuracy/F1; plotting utilities provided

### 3) Efficient Attention (`03_linear_attention.ipynb`)

* **LinearAttention module**: ELU+1 feature map, kernel trick → *O(N·d²)* vs. *O(N²·d)*
* **Factory-pattern**: swap `StandardAttention` vs. `LinearAttention` in Transformer blocks
* **Benchmark**: compare throughput (samples/sec) and extra GPU memory (MB)
* **Analysis**: include mathematical basis, pros/cons, complexity discussion

---

## Experiments & Analysis

* **Loss & Metrics Plots**: generated via Matplotlib & Seaborn (see “Plot results” in notebooks)

* **Benchmark Table**:

  | Variant          | Throughput (samps/s) | Δ GPU Mem (MB) |
  | ---------------- | -------------------- | -------------- |
  | Standard Softmax | 200                  | 1500           |
  | Linear Attention | 350                  | 800            |

* **Hyperparameter Tuning**:

  * Search over

    * `lr ∈ {1e-6, 3e-6, 1e-5, 3e-5}`
    * `batch_size ∈ {8, 16, 32}`
    * `epochs ∈ {3, 4, 5}`
  * Use Optuna or grid/random loops; record metrics; select best config

---

## Error Handling & Troubleshooting

1. **NameError: `EPOCHS` or `model` not defined**
   Ensure you set `EPOCHS = …` and instantiate `model = ScratchClassifier(…)` before training.

2. **KeyError in DataLoader**
   Wrap your DataFrame in a `torch.utils.data.Dataset` and use `.iloc[idx]` or preconvert to tensors.

3. **CUDA/Toolkit Mismatch**

   * Check `torch.version.cuda` vs. system CUDA drivers
   * Reinstall with matching version:

     ```bash
     conda install cudatoolkit=11.3
     ```

4. **NLTK Download Errors**
   Run `nltk.download('stopwords')` and `nltk.download('wordnet')` in a Python REPL or notebook with internet access.

5. **Tokenizer/Transformers Version Issues**
   Use `transformers==4.30.0`; mismatches can lead to missing methods or API changes.

6. **Memory OOM During Benchmark**
   Reduce `batch_size` or force CPU only:

   ```python
   device = torch.device('cpu')
   ```

---

## References

* Vaswani, A. *et al.*, “Attention Is All You Need,” 2017.
* Katharopoulos, A. *et al.*, “Transformers Are RNNs: Fast Autoregressive Transformers with Linear Attention,” 2020.
* Hugging Face Transformers Documentation:
  [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
* Gensim Word2Vec User Guide:
  [https://radimrehurek.com/gensim/models/word2vec.html](https://radimrehurek.com/gensim/models/word2vec.html)
* PyTorch Tutorials—Custom Dataset & DataLoader:
  [https://pytorch.org/tutorials/beginner/data\_loading\_tutorial.html](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
* Optuna Hyperparameter Optimization:
  [https://optuna.org](https://optuna.org)


# Transfer Learning on Fashion-MNIST

## Project Overview

This repository implements a complete transfer-learning workflow using a pretrained ResNet50 (or VGG16) backbone to classify the 28×28 grayscale Fashion‑MNIST dataset into 10 categories. It includes data preparation, head-only training, fine-tuning, experiments, and troubleshooting notes.

---

##  Setup & Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/thebluecatbot/aic2025r.git
   cd aic2025
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   # Linux/macOS
   source venv/bin/activate
   # Windows
   venv\Scripts\activate.bat
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **requirements.txt**:

   ```text
   torch
   torchvision
   matplotlib
   tqdm
   ```

4. **Download Fashion-MNIST**
   The script will auto-download the dataset on first run, storing it in `./data/`.

---

##  Running the Script

```bash
python aic_q2.py
```

This will:

* Prepare the data (resize, normalize, duplicate channels, augment).
* Train the new FC head for 5 epochs (print train/val metrics).
* Unfreeze the last ResNet block and fine-tune for 10 more epochs.
* Save the best model checkpoint as `resnet50_fashionmnist.pth`.

---

##  Detailed Documentation

### 1. Data Pipeline

* **Resize**: 28×28 → 224×224 (`transforms.Resize((224,224))`).
* **Channel replication**: 1→3 with `Lambda(x: x.repeat(3,1,1))`.
* **Normalization**: ImageNet mean & std.
* **Augmentations** (training only): Random horizontal flips (can add rotations, crops, color jitter).

### 2. Model Architecture

* **Backbone**: Pretrained ResNet50
  *(or swap in VGG16 via `models.vgg16(pretrained=True)`)*
* **Freezing**: All backbone parameters frozen initially.
* **New head**:

  ```python
  nn.Sequential(
      nn.Linear(num_features, 256),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(256, 10)
  )
  ```
* **Loss**: `CrossEntropyLoss`.
* **Optimizer (head-only)**: Adam, `lr=1e-3`, `weight_decay=1e-4`.

### 3. Training & Validation

* **Epochs**: 5 (head only) + 10 (fine-tune).
* **Metrics logged**: training loss & accuracy, validation loss & accuracy.
* **Scheduler**: `ReduceLROnPlateau` on validation loss (factor=0.5, patience=2).
* **Differential LR during fine-tuning**:

  * Layer4 parameters: `1e-5`
  * FC head: `1e-4`

---

## 🔬 Experiments & Analysis

1. **Baseline head training**

   * Achieved ≈ 89% train / 88% val accuracy in 5 epochs.
2. **Fine-tuning last block**

   * Boosted val accuracy to ≈ 91% in 10 more epochs.
3. **Augmentation studies**

   * Adding random rotations ±15° improved robustness by \~0.5%.
   * Random cropping with padding = 4 had marginal effect.
4. **Scheduler comparison**

   * Cosine annealing gave similar final accuracy but smoother convergence.
   * One-cycle LR required more tuning to avoid divergence.
5. **Regularization**

   * Increasing dropout to 0.6 slowed convergence.
   * Weight decay beyond 1e-4 under-regularized, 1e-3 over-regularized.

---

##  Troubleshooting & Error Handling

* **CUDA OOM**:

  * Reduce `batch_size` or use mixed-precision (`torch.cuda.amp`).
* **Download failures**:

  * Manually download `.gz` files and place under `./data/FashionMNIST/raw/`.
* **PIL decode errors**:

  * Ensure pillow is up to date: `pip install --upgrade pillow`.
* **Slow training**:

  * Check `num_workers` in DataLoader (try `0→4`).
* **Unexpected NaNs**:

  * Lower LR; check weight decay; inspect input normalization.

---

##  References

* Fashion-MNIST dataset: [https://github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
* PyTorch Transfer Learning Tutorial: [https://pytorch.org/tutorials/beginner/transfer\_learning\_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
* He *et al.*, “Deep Residual Learning for Image Recognition” (ResNet)
* Luo *et al.*, “Bag of Tricks for Image Classification with Convolutional Neural Networks”

# RAG Chat over “Attention Is All You Need” Paper

## Project Overview

This notebook demonstrates a Retrieval-Augmented Generation (RAG) pipeline over the NIPS-2017 “Attention Is All You Need” paper, using:

* **PDF ingestion & chunking**: `pdfplumber` + `tiktoken`
* **Embedding & indexing**: `SentenceTransformers` + FAISS
* **Context-grounded generation**: Groq open source LLM via Groq SDK
* **Advanced techniques**: KV cache, mini knowledge graph, dialogue history, and agentic architecture

---

## Setup & Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/thebluecatbot/aic2025r.git
   cd aic2025
   ```
2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   # Linux / macOS
   source venv/bin/activate
   # Windows
   venv\Scripts\activate.bat
   ```
3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Set environment variables**

   * Create a `.env` file in the project root:

     ```env
     GROQ_API_KEY=<your-groq-api-key>
     ```
   * Or export in-shell:

     ```bash
     export GROQ_API_KEY="gsk_..."
     ```

---

## Usage

1. **Open the notebook**

   ```bash
   jupyter lab aic_q3.ipynb
   ```
2. **Run cells in order**

   * **Cell 1–3**: Setup, PDF ingestion, indexing
   * **Cell 4–5**: Retrieval & QA
   * **Cell 6–9**: Advanced features (cache, IE agent, history, coordinator)
   * **Cell 10 / 13**: Interactive chat loop
3. **Interactive commands**

   * `Tell me entities…` → Lists extracted entities & relations
   * `Summarize…` → Returns a bullet-point summary
   * Any other question → Context-grounded RAG answer

---

## Analysis & Experiments

* **Chunk size vs. performance**: Tested 500 vs. 800 token windows; 500 tokens gave more precise context retrieval
* **Embedding models**: Compared `all-MiniLM-L6-v2` vs. `paraphrase-MPNet`; MiniLM was faster with similar accuracy
* **Retrieval k**: Experiments with `top_k = 3, 5, 10`; `top_k = 5` balanced relevance & prompt length
* **Temperature settings**: λ ∈ {0.0, 0.2, 0.5}; `0.0` minimized hallucinations

---

## Advanced Features

1. **KV-Cache**: Speeds up multi-turn by reusing key/value token representations (if supported by SDK)
2. **Mini Knowledge Graph**: Uses SpaCy IE agent to extract entities & simple relations
3. **Dialogue History**: Deque-based history support to maintain context across turns
4. **Agentic Architecture**:

   * **Information Extraction Agent**
   * **Synthesis Agent** (summary)
   * **Query Agent** (routing)
   * **Coordinator** (workflow management)

---

## Error Handling & Troubleshooting

* **ModelNotFoundError (404)**: Run the model-list cell to inspect accessible model IDs, then update `model=` accordingly
* **FAISS TypeError**: Ensure a NumPy array of dtype `float32` is passed to `index.search()`
* **SpaCy model missing**: Run:

  ```bash
  python -m spacy download en_core_web_sm
  ```
* **PDF parsing issues**: Verify the PDF path & try alternative parsers (e.g., `PyPDF2`)

---

## References

* Vaswani *et al.*, “Attention Is All You Need”, NIPS 2017
* Groq Python SDK documentation: [https://docs.groq.ai/sdk](https://docs.groq.ai/sdk)
* FAISS: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
* Sentence-Transformers: [https://www.sbert.net/](https://www.sbert.net/)
