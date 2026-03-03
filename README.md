# KWHHG_personality-graph

This repository implements the Knowledge-Enhanced Hierarchical Heterogeneous Graph (KEHHG) for Personality Identification, utilizing SKFRL for sentence splitting and a unique `UserBatchSampler` to prevent data-leakage during trait binarization.

---

## Environment Setup

Install all dependencies from the `requirements.txt` file (Generated securely):

```bash
pip install -r requirements.txt
```

*(Note: PyTorch CUDA is recommended. Ensure your local `torch` is compiled with CUDA support for the fastest `Doc-BERT` graph generation).*

---

## 🚀 How to Run the Pipeline (From Scratch)

Follow these 3 steps in order.

### Step 1: Preprocessing & Binarization
This step uses `SKFRL` (`sat-3l`) to split the raw dataset posts into sentences securely. It also correctly calculates the binarization mean thresholds **grouped by User ID** to prevent highly-talkative users from skewing the class labels.

```powershell
python -X utf8 src/preprocessing_pipeline.py
```
*Outputs: `final_train_preprocessed.csv`, `final_val_preprocessed.csv`, and `final_test_preprocessed.csv` inside `src/data/processed/preprocess_check_out/`.*

### Step 2: Global Graph Generation
This step builds the transductive adjacency matrices (Word, POS, LIWC/Empath, Entity, Text) and calculates the `Doc-BERT` embeddings using CUDA. **This step might take a few minutes.**

```powershell
set PYTHONPATH=src&& python -X utf8 run\run_global_graph.py
```
*Outputs: Pickled adjacency graphs (`adj_*.pkl`) and the main PyTorch geometric cache (`H_views.pt`) inside `src/outputs/global_graph_output/`.*

### Step 3: End-to-End Model Training
Finally, train the full KEHHG architecture. This script uses a custom `UserBatchSampler` to ensure that random batches contain grouped sentences from the *same users*. It also utilizes a harsh user-bias scaling mask (`same_user_bias=5.0`) before the softmax layer to enforce strict cross-sentence user attention.

```powershell
$env:TRAIN_CSV="src/data/processed/preprocess_check_out/final_train_preprocessed.csv"; $env:VAL_CSV="src/data/processed/preprocess_check_out/final_val_preprocessed.csv"; $env:TEST_CSV="src/data/processed/preprocess_check_out/final_test_preprocessed.csv"; python src/train_end2end.py
```
*Outputs: Prints step-gradient metrics and Saves `end2end_shine_best.pt` in `global_graph_output` upon completing the best validation epoch.*