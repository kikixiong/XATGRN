# XATGRN

This repository contains the implementation of the Cross-Attention Complex Dual Graph Embedding Model (XATGRN), a computational framework for inferring Gene Regulatory Networks (GRNs) . XATGRN features a cross-attention-based fusion module that extracts key details from gene expression profiles, enhancing gene representations and our understanding of gene-gene interactions. Its dual complex graph embedding in the relation graph module effectively tackles GRNs' skewed degree distribution, generating embeddings to accurately capture regulatory relationship.
## Environment Setup

```
conda env create -f duplexenvironment.yaml
```



```
conda env create -f grnenvironment.yaml
```




# 1. Graph Embedding

1. Activate the graph embedding environment `conda activate duplex`
2. Navigate to `./code/` directory
3. Excute the python `./train_edge/train.py` 



# 2. GRN prediction

1.Activate the graph GRN prediction environment `conda activate grn`

### Shell excution

```
bash
chmod +x run_all_datasets.sh

./run_all_datasets.sh

```
### Manual excution
There are 9 datasets to be evaluated, and each dataset script needs to be executed manually.

**example usage for DREAM5net1 datasets**

```
cd Prediction
cd DREAM5_net1_FGRN
python DREAM5net1_FCV.py
```

Finally, check output folder "results" for results. The csv file shows the mean and standard deviation of AUROC, MCC, F1, Recall, Precision  on this dataset.







