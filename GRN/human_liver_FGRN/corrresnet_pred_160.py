import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.utils import class_weight
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau



class CAN_Layer(nn.Module):
    def __init__(self, hidden_dim, num_heads, group_size, agg_mode,num_positions=805):
        super(CAN_Layer, self).__init__()
        self.agg_mode = agg_mode
        self.group_size = group_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads

        self.query_p = nn.LazyLinear(hidden_dim, bias=False)
        self.key_p = nn.LazyLinear(hidden_dim, bias=False)
        self.value_p = nn.LazyLinear(hidden_dim, bias=False)

        self.query_d = nn.LazyLinear(hidden_dim, bias=False)
        self.key_d = nn.LazyLinear(hidden_dim, bias=False)
        self.value_d = nn.LazyLinear(hidden_dim, bias=False)



    def alpha_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape
        mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
        mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)
        mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col)

        logits = torch.where(mask_pair, logits, logits - inf)
        alpha = torch.softmax(logits, dim=2)
        mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
        alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
        return alpha

    def apply_heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def group_embeddings(self, x, mask, group_size):
        N, L, D = x.shape
        groups = L // group_size
        x_grouped = x.view(N, groups, group_size, D).mean(dim=2)
        mask_grouped = mask.view(N, groups, group_size).any(dim=2)
        return x_grouped, mask_grouped
    

    def forward(self, protein, drug, mask_prot, mask_drug):
        # Add an additional dimension to handle 2D inputs
        protein = protein.unsqueeze(1)  # [batch_size, 1, dim]
        drug = drug.unsqueeze(1)  # [batch_size, 1, dim]
        
        # Create dummy masks since we have only one "sequence" element
        mask_prot = torch.ones(protein.size()[:-1], dtype=torch.bool, device=protein.device)
        mask_drug = torch.ones(drug.size()[:-1], dtype=torch.bool, device=drug.device)

        # Compute queries, keys, values for both protein and drug after grouping
        query_prot = self.apply_heads(self.query_p(protein), self.num_heads, self.head_size)# [batch_size, 1, num_heads, head_size]
        key_prot = self.apply_heads(self.key_p(protein), self.num_heads, self.head_size)
        value_prot = self.apply_heads(self.value_p(protein), self.num_heads, self.head_size)

        query_drug = self.apply_heads(self.query_d(drug), self.num_heads, self.head_size)
        key_drug = self.apply_heads(self.key_d(drug), self.num_heads, self.head_size)
        value_drug = self.apply_heads(self.value_d(drug), self.num_heads, self.head_size)


        # Compute attention scores
        logits_pp = torch.einsum('blhd, bkhd->blkh', query_prot, key_prot)
        logits_pd = torch.einsum('blhd, bkhd->blkh', query_prot, key_drug)
        logits_dp = torch.einsum('blhd, bkhd->blkh', query_drug, key_prot)
        logits_dd = torch.einsum('blhd, bkhd->blkh', query_drug, key_drug)

        alpha_pp = self.alpha_logits(logits_pp, mask_prot, mask_prot)
        alpha_pd = self.alpha_logits(logits_pd, mask_prot, mask_drug)
        alpha_dp = self.alpha_logits(logits_dp, mask_drug, mask_prot)
        alpha_dd = self.alpha_logits(logits_dd, mask_drug, mask_drug)

        prot_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_pp, value_prot).flatten(-2) +
                          torch.einsum('blkh, bkhd->blhd', alpha_pd, value_drug).flatten(-2)) / 2
        drug_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_dp, value_prot).flatten(-2) +
                          torch.einsum('blkh, bkhd->blhd', alpha_dd, value_drug).flatten(-2)) / 2

        if self.agg_mode == "cls":
            prot_embed = prot_embedding[:, 0]  # query : [batch_size, hidden]
            drug_embed = drug_embedding[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            prot_embed = prot_embedding.mean(1)  # query : [batch_size, hidden]
            drug_embed = drug_embedding.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            prot_embed = (prot_embedding * mask_prot.unsqueeze(-1)).sum(1) / mask_prot.sum(-1).unsqueeze(-1)
            drug_embed = (drug_embedding * mask_drug.unsqueeze(-1)).sum(1) / mask_drug.sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()

        query_embed = torch.cat([prot_embed, drug_embed], dim=1)
        return query_embed





def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

class Classifier(nn.Module):
    def __init__(self, args,output_directory, nb_classes, x_tarin1, x_tarin2, net_emd_tf_s, net_emd_tf_t, net_emd_target_s, net_emd_target_t, verbose=False, build=True, load_weights=False, patience=5):
        super(Classifier, self).__init__()
        self.patience = 5
        self.output_directory = output_directory
        # self.feature_model1 = FeatureModel(x_tarin1)
        # self.feature_model2 = FeatureModel(x_tarin2)


        self.can_layer = CAN_Layer(hidden_dim=128, num_heads=8, group_size=args.group_size, agg_mode=args.agg_mode)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        # self.conv = nn.Conv1d(640, 16, 16, padding=8)

        # Linear layer to match Transformer input dimension
        self.fc_transformer_input = nn.Linear(640, 768)

        self.conv = nn.Conv1d(640, 16, 15, padding=7)
        self.bn = nn.BatchNorm1d(16, momentum=0.8)
        self.pool = nn.MaxPool1d(5, padding=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc_pred2 = nn.Linear(16, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc_final = nn.Linear(128, nb_classes)


    def forward(self, x_train1, x_train2, net_emd_tf_s, net_emd_tf_t, net_emd_target_s, net_emd_target_t):


        f1 = x_train1.squeeze(2)
        f2 = x_train2.squeeze(2)
        # f1 = self.embedding1(x1)
        # f2 = self.embedding2(x2)
        # print(f'f1 size{f1.shape}')
        mask_f1 = torch.ones(f1.size()[:-1], dtype=torch.bool, device=f1.device)
        mask_f2 = torch.ones(f2.size()[:-1], dtype=torch.bool, device=f2.device)
        # print(f'mask size{mask_f1.shape}')
        
        combined_features = self.can_layer(f1, f2, mask_f1, mask_f2)
        
        combined_features = F.relu(self.fc1(combined_features))
        combined_features = self.dropout(combined_features)
        combined_features = F.relu(self.fc2(combined_features))
        # out = self.fc_pred2(combined_features)
        # print(f'combined feature size{combined_features.shape}')
        input_layer_net_tf_s_ = net_emd_tf_s.squeeze(1)
        input_layer_net_tf_t_ = net_emd_tf_t.squeeze(1)
        input_layer_net_target_s_ = net_emd_target_s.squeeze(1)
        input_layer_net_target_t_ = net_emd_target_t.squeeze(1)
            # print(f'net_emd_tf_s_train shape: {input_layer_net_tf_s_.shape}')
            # print(f'net_emd_tf_t_train shape: {input_layer_net_tf_t_.shape}')
            # print(f'net_emd_target_s_train shape: {input_layer_net_target_s_.shape}')
            # print(f'net_emd_target_t_train shape: {input_layer_net_target_t_.shape}')

            # Concatenate features
        all_features = torch.cat(
            [combined_features, input_layer_net_tf_s_, input_layer_net_tf_t_, input_layer_net_target_s_,
                 input_layer_net_target_t_], dim=1)
        # print(f'all_features shape: {all_features.shape}')

            # Reshape
        all_features = all_features.unsqueeze(1).permute(0,2,1)

            # Convolution and Batch Normalization
        x = self.conv(all_features)
        x = self.bn(x)
        x = self.pool(x)
        x = F.relu(x)

            # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for dense layer

            # Fully connected layers
        x = F.relu(self.fc_pred2(x))
        x = self.dropout(x)
        x = self.fc_final(x)
        # print(f'output shape: {x.shape}')
        return x

    def fit_5CV(self, x_train_1, x_train_2, net_emd_tf_s_train, net_emd_tf_t_train, net_emd_target_s_train,net_emd_target_t_train, y_train,
                x_val_1, x_val_2,net_emd_tf_s_val, net_emd_tf_t_val, net_emd_target_s_val,net_emd_target_t_val, y_val,
                x_test_1, x_test_2,net_emd_tf_s_test, net_emd_tf_t_test, net_emd_target_s_test,net_emd_target_t_test):

        batch_size = 128
        # nb_epochs = 150
        nb_epochs = 80

        y_train_num = []
        for i in range(y_train.shape[0]):
            a = y_train[i][0]
            b = y_train[i][1]
            c = y_train[i][2]

            if a == 1:
                y_train_num.append(0)
            elif b == 1:
                y_train_num.append(1)
            elif c == 1:
                y_train_num.append(2)
            else:
                print('error y-train')
        y_train_num = np.array(y_train_num)
        class_weights = class_weight.compute_class_weight("balanced", np.unique(y_train_num), y_train_num)
        print(class_weights)
        print('------------------------------------------------------------------------------')
        mini_batch_size = int(min(x_train_1.shape[0] / 10, batch_size))
                # Print dimensions before training loop
        print(f'x_train_1 shape: {x_train_1.shape}')
        print(f'x_train_2 shape: {x_train_2.shape}')
        print(f'net_emd_tf_s_train shape: {net_emd_tf_s_train.shape}')
        print(f'net_emd_tf_t_train shape: {net_emd_tf_t_train.shape}')
        print(f'net_emd_target_s_train shape: {net_emd_target_s_train.shape}')
        print(f'net_emd_target_t_train shape: {net_emd_target_t_train.shape}')
        print(f'y_train shape: {y_train.shape}')



        # ======================================== pytorch-train =============================================

        train_dataset = TensorDataset(x_train_1, x_train_2, net_emd_tf_s_train, net_emd_tf_t_train,
                                      net_emd_target_s_train, net_emd_target_t_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True)

        val_dataset = TensorDataset(x_val_1, x_val_2, net_emd_tf_s_val, net_emd_tf_t_val, net_emd_target_s_val,
                                    net_emd_target_t_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=mini_batch_size, shuffle=False)

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.001, patience=int(self.patience / 2), verbose=True)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))
        # criterion = nn.CrossEntropyLoss()


        patience = 0
        best_val_acc = 0
        last_acc = 0
        for epoch in range(nb_epochs):
            self.train()
            total_loss = 0
            for inputs in train_loader:
                # Unpack your inputs depending on your specific needs
                x1, x2, emd_s, emd_t, target_s, target_t, labels = inputs
                optimizer.zero_grad()
                outputs = self(x1.permute(0,2,1), x2.permute(0,2,1), emd_s, emd_t, target_s, target_t)
                _, labels = torch.max(labels, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch: {epoch}, Batch: {i}, Current LR: {current_lr}')

            # 验证阶段
            self.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for inputs in val_loader:
                    x1, x2, emd_s, emd_t, target_s, target_t, labels = inputs
                    outputs = self(x1.permute(0,2,1), x2.permute(0,2,1), emd_s, emd_t, target_s, target_t)
                    _, labels = torch.max(labels, 1)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = correct / total
            # scheduler.step(val_acc)
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.state_dict(), self.output_directory+'best_model.pth')
            #     patience = 0
            # elif val_acc > last_acc:
            #     patience -= 1
            # else:
            #     patience += 1
            #     if patience >= self.patience:
            #         print("Early stopping due to no improvement")
            #         break
            # last_acc = val_acc
            print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)},Val Accuracy: {val_acc}')
        y_pred = self.predict(
            x_test_1, x_test_2, net_emd_tf_s_test, net_emd_tf_t_test, net_emd_target_s_test, net_emd_target_t_test)
        yy_pred = np.argmax(y_pred, axis=1)
        return y_pred, yy_pred

    def predict(self, x_test_1, x_test_2,net_emd_tf_s_test, net_emd_tf_t_test, net_emd_target_s_test,net_emd_target_t_test):
        test_dataset = TensorDataset(x_test_1, x_test_2,net_emd_tf_s_test, net_emd_tf_t_test, net_emd_target_s_test,net_emd_target_t_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        # Load the best model
        self.load_state_dict(torch.load(f'{self.output_directory}/best_model.pth'))
        # Test the model
        self.eval()
        predictions = []
        with torch.no_grad():
            for inputs in test_loader:
                x1, x2, emd_s, emd_t, target_s, target_t = inputs
                predicted = self(x1.permute(0, 2, 1), x2.permute(0, 2, 1), emd_s, emd_t, target_s, target_t)
                predictions.extend(predicted.cpu().numpy())
        return predictions