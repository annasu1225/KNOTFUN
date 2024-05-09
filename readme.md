# Improved Protein Function Prediction by Combining Persistent Cohomology and ProteinBERT Embeddings

Below are two options to use our model:

* **Option 1 - Reproduce our Results** Use our human protein dataset which contains 6130 *H. Sapiens* proteins scraped from the Protein Data Bank (PDB) using the following parameters: resolution $\leq$ 1.5 Å, solved with X-Ray diffraction, of length $\leq$ 512, and published 2020 or later, and corresponding molecular function Gene Ontology (GO) terms from UniProt, with a total of 120 unique functional annotations.

* **Option 2 - Bring your own dataset** With your PDBs, labels, and sequences, train PersLay from scratch, combine the embeddings with pretrained ProteinBERT, and train a final classification layer combining the outputs of the two models using our pipeline.

## Option 1 - Start From Our Dataset & Reproduce Results

1. Download our dataset & model weights from (here)[]

    * x_data.npy
    * y_data.npy
    * labels.csv
    * best_model

2. Use `main.py` to reproduce our results and ablations

```
main.py --x_file x_data.npy --y_file y_data.npy --labels_file labels.csv --model_path best_model
```

* Use `--pb_only` for ProteinBERT only
* Use `--ph_only` for PersLay only
* Use `--ablation` to reproduce our ablation study
* Exclude `--model_path` to reproduce training \& hyperparameter tuning

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
