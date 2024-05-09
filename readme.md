# Improved Protein Function Prediction by Combining Persistent Cohomology and ProteinBERT Embeddings

Below are two options to use our model:

* **Option 1 - Reproduce our Results** Use our human protein dataset which contains 6130 *H. Sapiens* proteins scraped from the Protein Data Bank (PDB) using the following parameters: resolution $\leq$ 1.5 Å, solved with X-Ray diffraction, of length $\leq$ 512, and published 2020 or later, and corresponding molecular function Gene Ontology (GO) terms from UniProt, with a total of 120 unique functional annotations.

* **Option 2 - Bring your own dataset** With your PDBs, labels, and sequences, train PersLay from scratch, combine the embeddings with pretrained ProteinBERT, and train a final classification layer combining the outputs of the two models using our pipeline.

## Option 1 - Start From Our Dataset & Reproduce Results

1. Download our dataset & model weights from (here)[]

    * x_data.npy
    * y_data.npy
    * labels.npy
    * best_model

2. Use `main.py` to reproduce our results and ablations

`--x_file $YOUR_DOWNLOAD_PATH --y_file $YOUR_DOWNLOAD_PATH --labels_file $YOUR_DOWNLOAD_PATH --model_path $YOUR_DOWNLOAD_PATH --ablation --top_k 120`

train_tf.py -> GitHub
Use —pb_only for PB Only
Use —ph_only for PH Only

Reproduce hyperparameter tubing
Don’t set —hyperparamters

Reproduce Ablations
—ablations —hyperparams —model “”

Ablations


Option 2
Start from any PH and PB embeddings

PB CSV Example-> Drive
PH CSV Example-> Drive

PH Folder Structure Example

PB Folder Structure Example

make_ph.py 
make_pb.py 

train.py —from_scratch

Note: All training was done on NVIDIA A100 details….
