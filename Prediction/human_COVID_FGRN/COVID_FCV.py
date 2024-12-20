import pandas as pd
import numpy as np
import random, os,time

import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score,recall_score,precision_score,matthews_corrcoef
import three_utils_2_single
import corrresnet_pred_224 as corrresnet_pred
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument(
        "--agg_mode", default="mean_all_tok", type=str, help="{cls|mean|mean_all_tok}"
    )
    parser.add_argument("--group_size", type=int, default=1)

    return parser.parse_args()
args = parse_config()



iteration = 10
nb_classes = 3
num_not_regulate = 6000

dim_net = 128
dim_exp = 42
a = dim_exp
b = a + dim_exp
c = b + dim_net
d = c + dim_net
e = d + dim_net
f = e + dim_net

path_network_name_type = 'traindataHuman/final_GRN/new_GRN_COVID_GEN_counts_genename.csv'

path_expression = 'traindataHuman/final_expression/COVID_GEN_counts.csv'
path_network_ids = 'traindataHuman/final_GRN_ids/new_GRN_COVID_GEN_counts_genename_ids.tsv'
path_node = 'traindataHuman/final_genelist_txt/new_exp_COVID_GEN_counts_genename_ids.txt'

output_directory = './output_directory/'
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)
logTime = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
network_dict_name = 'COVIDS_counts_'  + str(dim_net) + '_' + logTime
save_index_path = './results/'
if not os.path.isdir(save_index_path):
    os.makedirs(save_index_path)

EXP_cold_raw = pd.read_csv(path_expression, sep='\,', header=None,engine='python')
# 剔除最后两行
EXP_cold_raw = EXP_cold_raw.iloc[:-2]
EXP_cold = EXP_cold_raw.loc[1:,1:]
EXP_cold = np.array(EXP_cold)
EXP_cold_new = np.zeros((EXP_cold.shape[0],EXP_cold.shape[1]))
for i in range(EXP_cold.shape[0]):
    for j in range(EXP_cold.shape[1]):
        EXP_cold_new[i][j] = float(EXP_cold[i][j])

genename = EXP_cold_raw.loc[1:,0]
genename = np.array(genename)

Ecoli_GRN_known = pd.read_csv(path_network_name_type, sep='\,', header=None,engine='python')

Ecoli_GRN, num_activator, num_repressor, num_unknown = three_utils_2_single.get_GRN(Ecoli_GRN_known,genename)

# 存储最高指标及对应文件名的变量
best_scores = {
    "AUROC_mean": {"value": 0, "file_name": ""},
    "Recall_mean": {"value": 0, "file_name": ""},
    "Precision_mean": {"value": 0, "file_name": ""},
    "F1_mean": {"value": 0, "file_name": ""},
    "MCC_mean": {"value": 0, "file_name": ""}
}


# 遍历duplex目录下的所有npz文件
# 使用相对路径
base_dir = os.path.dirname(__file__)  # 获取当前脚本所在的目录
duplex_directory = os.path.join(base_dir, '..', '..', 'DUPLEX-master', 'saveGraphEmbedding', 'human')
for file_name in os.listdir(duplex_directory):
    if file_name.endswith('.npz'):
        npz_path = os.path.join(duplex_directory, file_name)
        data = np.load(npz_path)
        GRN_embedding_s = data['am_embedding']
        GRN_embedding_t = data['ph_embedding']
        print(f"Processing {file_name} - Shape of GRN_embedding_s: {GRN_embedding_s.shape}, GRN_embedding_t: {GRN_embedding_t.shape}")

        # 10次5折交叉验证
        network_dict = {
            "AUROC mean": 0, "AUROC std": 0,
            "Recall mean": 0, "Recall std": 0,
            "Precision mean": 0, "Precision std": 0,
            "F1 mean": 0, "F1 std": 0,
            "MCC mean": 0, "MCC std": 0
        }
        all_network_dict = {
            "AUROC": 0, "Recall": 0,
            "Precision": 0, "F1": 0, "MCC": 0
        }
        kf = KFold(n_splits=5, shuffle=True)
        netavgAUROCs = []
        netavgRecalls = []
        netavgPrecisions = []
        netavgF1s = []
        netavgMCCs = []

        for ki in range(iteration):
            print("\nthe {}th five-fold cross-validation..........\n".format(ki + 1))

            positive2_data, positive1_data, negative0_data, feature_size_tf, feature_size_target, feature_size_tf_nets = three_utils_2_single.create_samples_human_counts(
                EXP_cold_new, Ecoli_GRN, GRN_embedding_s, GRN_embedding_t, num_not_regulate)

            alldata = np.vstack((positive2_data, positive1_data))
            alldata = np.vstack((alldata, negative0_data))
            np.random.shuffle(alldata)

            dataX_tf, dataX_target, net_tf_s, net_tf_t, net_target_s, net_target_t, labelY, position = three_utils_2_single.transform_data_single_net(alldata)

            AUROCs = []
            Recalls = []
            Precisions = []
            F1s = []
            MCCs = []

            dataX = []
            for i in range(dataX_tf.shape[0]):
                temp = np.hstack((dataX_tf[i], dataX_target[i]))
                temp = np.hstack((temp, net_tf_s[i]))
                temp = np.hstack((temp, net_tf_t[i]))
                temp = np.hstack((temp, net_target_s[i]))
                temp = np.hstack((temp, net_target_t[i]))
                dataX.append(temp)

            dataX = np.array(dataX)
            for train_index, test_index in kf.split(dataX, labelY):
                trainX, testX = dataX[train_index], dataX[test_index]
                trainY, testY = labelY[train_index], labelY[test_index]

                trainXX, testXX, trainYY, testYY = train_test_split(trainX, trainY, test_size=0.2, random_state=1, shuffle=True)

                trainXX_tf = torch.from_numpy(trainXX[:, :, 0:a]).float()
                trainXX_target = torch.from_numpy(trainXX[:, :, a:b]).float()
                trainXX_net_tf_s = torch.from_numpy(trainXX[:, :, b:c]).float()
                trainXX_net_tf_t = torch.from_numpy(trainXX[:, :, c:d]).float()
                trainXX_net_target_s = torch.from_numpy(trainXX[:, :, d:e]).float()
                trainXX_net_target_t = torch.from_numpy(trainXX[:, :, e:f]).float()

                testXX_tf = torch.from_numpy(testXX[:, :, 0:a]).float()
                testXX_target = torch.from_numpy(testXX[:, :, a:b]).float()
                testXX_net_tf_s = torch.from_numpy(testXX[:, :, b:c]).float()
                testXX_net_tf_t = torch.from_numpy(testXX[:, :, c:d]).float()
                testXX_net_target_s = torch.from_numpy(testXX[:, :, d:e]).float()
                testXX_net_target_t = torch.from_numpy(testXX[:, :, e:f]).float()

                testX_tf = torch.from_numpy(testX[:, :, 0:a]).float()
                testX_target = torch.from_numpy(testX[:, :, a:b]).float()
                testX_net_tf_s = torch.from_numpy(testX[:, :, b:c]).float()
                testX_net_tf_t = torch.from_numpy(testX[:, :, c:d]).float()
                testX_net_target_s = torch.from_numpy(testX[:, :, d:e]).float()
                testX_net_target_t = torch.from_numpy(testX[:, :, e:f]).float()

                classifier = corrresnet_pred.Classifier(args, output_directory, nb_classes, trainXX_tf, trainXX_target, trainXX_net_tf_s, trainXX_net_tf_t, trainXX_net_target_s, trainXX_net_target_t, verbose=True, patience=5)

                score_1, score_int = classifier.fit_5CV(trainXX_tf, trainXX_target, trainXX_net_tf_s, trainXX_net_tf_t, trainXX_net_target_s, trainXX_net_target_t, trainYY, testXX_tf, testXX_target, testXX_net_tf_s, testXX_net_tf_t, testXX_net_target_s, testXX_net_target_t, testYY, testX_tf, testX_target, testX_net_tf_s, testX_net_tf_t, testX_net_target_s, testX_net_target_t)

                testY_int = np.argmax(testY, axis=1)

                AUC = roc_auc_score(testY, score_1, multi_class='ovo')
                Recall = recall_score(testY_int, score_int, average='weighted')
                Precision = precision_score(testY_int, score_int, average='weighted')
                F1 = f1_score(testY_int, score_int, average='weighted')
                MCC = matthews_corrcoef(testY_int, score_int)

                AUROCs.append(AUC)
                Recalls.append(Recall)
                Precisions.append(Precision)
                F1s.append(F1)
                MCCs.append(MCC)

            avg_AUROC = np.mean(AUROCs)
            avg_Recalls = np.mean(Recalls)
            avg_Precisions = np.mean(Precisions)
            avg_F1s = np.mean(F1s)
            avg_MCCs = np.mean(MCCs)
            print("\nAUROC of the {}th five-fold cross-validation is \n".format(ki + 1) + str(avg_AUROC))
            print("\nMCC of the {}th five-fold cross-validation is \n".format(ki + 1) + str(avg_MCCs))
            print("\nRecall of the {}th five-fold cross-validation is \n".format(ki + 1) + str(avg_Recalls))
            print("\nPrecision of the {}th five-fold cross-validation is \n".format(ki + 1) + str(avg_Precisions))
            print("\nF1 of the {}th five-fold cross-validation is \n".format(ki + 1) + str(avg_F1s))

            netavgAUROCs.append(avg_AUROC)
            netavgRecalls.append(avg_Recalls)
            netavgPrecisions.append(avg_Precisions)
            netavgF1s.append(avg_F1s)
            netavgMCCs.append(avg_MCCs)

        # 计算整体的平均和标准差
        AUROC_mean = np.mean(netavgAUROCs)
        AUROC_std = np.std(netavgAUROCs, ddof=1)
        Recall_mean = np.mean(netavgRecalls)
        Recall_std = np.std(netavgRecalls)
        Precision_mean = np.mean(netavgPrecisions)
        Precision_std = np.std(netavgPrecisions)
        F1_mean = np.mean(netavgF1s)
        F1_std = np.std(netavgF1s)
        MCC_mean = np.mean(netavgMCCs)
        MCC_std = np.std(netavgMCCs)

        # 保存每个指标的均值和标准差
        network_dict["AUROC mean"] = float('{:.4f}'.format(AUROC_mean))
        network_dict["AUROC std"] = float('{:.4f}'.format(AUROC_std))
        network_dict["Recall mean"] = float('{:.4f}'.format(Recall_mean))
        network_dict["Recall std"] = float('{:.4f}'.format(Recall_std))
        network_dict["Precision mean"] = float('{:.4f}'.format(Precision_mean))
        network_dict["Precision std"] = float('{:.4f}'.format(Precision_std))
        network_dict["F1 mean"] = float('{:.4f}'.format(F1_mean))
        network_dict["F1 std"] = float('{:.4f}'.format(F1_std))
        network_dict["MCC mean"] = float('{:.4f}'.format(MCC_mean))
        network_dict["MCC std"] = float('{:.4f}'.format(MCC_std))

        # 更新最佳文件的记录
        if AUROC_mean > best_scores["AUROC_mean"]["value"]:
            best_scores["AUROC_mean"]["value"] = AUROC_mean
            best_scores["AUROC_mean"]["file_name"] = file_name
        if Recall_mean > best_scores["Recall_mean"]["value"]:
            best_scores["Recall_mean"]["value"] = Recall_mean
            best_scores["Recall_mean"]["file_name"] = file_name
        if Precision_mean > best_scores["Precision_mean"]["value"]:
            best_scores["Precision_mean"]["value"] = Precision_mean
            best_scores["Precision_mean"]["file_name"] = file_name
        if F1_mean > best_scores["F1_mean"]["value"]:
            best_scores["F1_mean"]["value"] = F1_mean
            best_scores["F1_mean"]["file_name"] = file_name
        if MCC_mean > best_scores["MCC_mean"]["value"]:
            best_scores["MCC_mean"]["value"] = MCC_mean
            best_scores["MCC_mean"]["file_name"] = file_name


        unique_network_dict_name = network_dict_name + '_' + file_name.replace('.npz', '')

        with open(save_index_path + unique_network_dict_name + '_avg.csv', 'w') as filename:
            for k, v in network_dict.items():
                filename.write(k + ':' + str(v))
                filename.write('\n')
            filename.close()

        all_network_dict["AUROC"] = netavgAUROCs
        all_network_dict["Recall"] = netavgRecalls
        all_network_dict["Precision"] = netavgPrecisions
        all_network_dict["F1"] = netavgF1s
        all_network_dict["MCC"] = netavgMCCs

        with open(save_index_path + unique_network_dict_name + '_all.csv', 'w') as filename:
            for k, v in all_network_dict.items():
                filename.write(k + ':' + str(v))
                filename.write('\n')
            filename.close()

# 将最高指标及对应文件名保存到文本文件
with open(save_index_path + 'best_scores.txt', 'w') as txt_file:
    txt_file.write(f"Highest AUROC_mean: {best_scores['AUROC_mean']['value']} from file {best_scores['AUROC_mean']['file_name']}\n")
    txt_file.write(f"Highest Recall_mean: {best_scores['Recall_mean']['value']} from file {best_scores['Recall_mean']['file_name']}\n")
    txt_file.write(f"Highest Precision_mean: {best_scores['Precision_mean']['value']} from file {best_scores['Precision_mean']['file_name']}\n")
    txt_file.write(f"Highest F1_mean: {best_scores['F1_mean']['value']} from file {best_scores['F1_mean']['file_name']}\n")
    txt_file.write(f"Highest MCC_mean: {best_scores['MCC_mean']['value']} from file {best_scores['MCC_mean']['file_name']}\n")


# 打印出各指标中mean最高的文件名
print(f"Highest AUROC_mean: {best_scores['AUROC_mean']['value']} from file {best_scores['AUROC_mean']['file_name']}")
print(f"Highest Recall_mean: {best_scores['Recall_mean']['value']} from file {best_scores['Recall_mean']['file_name']}")
print(f"Highest Precision_mean: {best_scores['Precision_mean']['value']} from file {best_scores['Precision_mean']['file_name']}")
print(f"Highest F1_mean: {best_scores['F1_mean']['value']} from file {best_scores['F1_mean']['file_name']}")
print(f"Highest MCC_mean: {best_scores['MCC_mean']['value']} from file {best_scores['MCC_mean']['file_name']}")


