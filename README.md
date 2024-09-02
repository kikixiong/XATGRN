# GRN

由于两个过程中的torch和dgl的版本冲突原因需要配两个环境、其中一个是用于DUPLEX进行graph embedding，分别都导出了`requirements.txt`和`environment.yml`以便配置



## Environment Setup

```
conda env create -f environment.yml
```

or

```
pip install requirements.txt
```



# Graph Embedding

1. Navigate to `./code/` directory
2. Excute the python `./train_edge/train.py`



## GRN prediction

There are 9 datasets to be evaluated, and each dataset script needs to be executed manually.

**example usage for DREAM5net1 datasets**

```
cd GRN
cd DREAM5_net1_FGRN
python DeepFGRN_DREAM5net1_FCV.py
```

Finally, check output folder "results" for results. The csv file shows the mean and standard deviation of AUROC, MCC, F1, Recall, Precision  on this dataset.





