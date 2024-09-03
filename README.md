# GRN

由于两个过程中的torch和dgl的版本冲突原因需要配两个环境、其中一个是用于DUPLEX进行graph embedding，分别在GRN和DUPLEX-master文件夹中都导出了对应环境的配置`requirements.txt`



## Environment Setup

```
pip install requirements.txt
```



# 1. Graph Embedding

1. Activate the graph embedding environment `conda activate environment1`
2. Navigate to `./code/` directory
3. Excute the python `./train_edge/train.py` 



# 2. GRN prediction

1.Activate the graph GRN prediction environment `conda activate environment2`

### Shell excution
**确保脚本有执行权限并执行脚本**：

```
bash
chmod +x run_all_datasets.sh

./run_all_datasets.sh

```
### Manual excution
There are 9 datasets to be evaluated, and each dataset script needs to be executed manually.

**example usage for DREAM5net1 datasets**

```
cd GRN
cd DREAM5_net1_FGRN
python DeepFGRN_DREAM5net1_FCV.py
```

Finally, check output folder "results" for results. The csv file shows the mean and standard deviation of AUROC, MCC, F1, Recall, Precision  on this dataset.







