# Improved Protein Function Prediction by Combining Persistent Cohomology and ProteinBERT Embeddings

Below are two options to use our model:

* **Option 1 - Reproduce our Results** Use our human protein dataset which contains 6130 *H. Sapiens* proteins scraped from the Protein Data Bank (PDB) using the following parameters: resolution $\leq$ 1.5 Å, solved with X-Ray diffraction, of length $\leq$ 512, and published 2020 or later, and corresponding molecular function Gene Ontology (GO) terms from UniProt, with a total of 120 unique functional annotations.

* **Option 2 - Bring your own dataset** With your PDBs, labels, and sequences, train PersLay from scratch, combine the embeddings with pretrained ProteinBERT, and train a final classification layer combining the outputs of the two models using our pipeline.

## Option 1 - Start From Our Dataset & Reproduce Results

1. Download our dataset & model weights from [here](https://drive.google.com/drive/folders/1vwBZ9MLocZCF6WLP4ZgvAZ3Fj2Bi117v?usp=drive_link).

    * x_data.npy
    * y_data.npy
    * labels.csv
    * best_model

2. Use `main.py` to reproduce our results and ablations after installing dependencies listed below under

```
main.py --x_file x_data.npy --y_file y_data.npy --labels_file labels.csv --model_path best_model
```

* Use `--pb_only` for ProteinBERT only
* Use `--ph_only` for PersLay only
* Use `--ablation` to reproduce our ablation study
* Exclude `--model_path` to reproduce training \& hyperparameter tuning

## Option 2 - Bring Your Own Dataset

1. Install dependencies under **PersLay Training** and **Ripser++**
2. Generate Vietoris–Rips persistence barcodes for your data using `sh calculate_ph/run_ph.sh`, modifying `input_dir` and `output_dir` to your directories
3. Train PersLay on your dataset and generate features using

```
python calculate_ph/ph_functions_h2_v2.py input_dir output_dir
```

4. Create a new environment and install the dependencies under **PersLay+ProteinBERT Training**
5. Clone the [ProteinBERT Repo](https://github.com/nadavbra/protein_bert) and download their weights. Follow the instructions on their repository to generate ProteinBERT embeddings.
6. Use `main.py` with the `--from_scratch` flag as follows
```
python main.py --from_scratch --ph_path your_directory --labels_path your_directory --pb_file your_directory
```

* After running with `from_scratch`, the `X.npy` and `Y.npy` files will be automatically generated and can be used with `--x_file` and `--y_file` in future runs
* The `--ablation` flag can be set to also run the ablation study

## Dependencies

### For Ripser++
- ripserplusplus (1.1.3)

### PersLay Training
- numpy (1.24.3)
- tensorflow (2.13.0)
- sklearn (1.3.2)
- scipy (1.11.4)
- pandas (2.2.0)
- matplotlib-base (3.8.2)
- h5py (3.10.0)
- guhdi (3.8.0)

### PersLay+ProteinBERT Training
- tensorflow (2.4.0)
- tensorflow_addons (0.12.1)
- numpy (1.20.1)
- pandas (1.2.3)
- h5py (3.11.0)
- lxml (5.2.1)
- pyfaidx (0.8.1.1)

## References
1. Zhang S, Xiao M, Wang H. GPU-Accelerated Computation of Vietoris-Rips Persistence Barcodes. 2020 Mar 17. [GitHub](https://github.com/simonzhang00/ripser-plusplus/tree/master)
2. Carrière M, Chazal F, Ike Y, Lacombe T, Royer M, Umeda Y. PersLay: A Neural Network Layer for Persistence Diagrams and New Graph Topological Signatures. Proc Mach Learn Res. 2020;108:2786–96. [GitHub](https://github.com/MathieuCarriere/perslay)
3. Brandes N, Ofer D, Peleg Y, Rappoport N, Linial M. ProteinBERT: a universal deep-learning model of protein sequence and function. Martelli PL, editor. Bioinformatics. 2022 Apr 12;38(8):2102–10. [GitHub](https://github.com/nadavbra/protein_bert)
   
