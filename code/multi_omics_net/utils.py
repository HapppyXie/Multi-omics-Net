from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch

# 展示模型性能
def display_performance(best_model, X_test, y_test,feature_dims):
    best_model.eval()
    with torch.no_grad():   
        # 计算测试损失
        pred_probability = best_model(
            [
                X_test[:, 0, :feature_dims[0]],
                X_test[:, 1, :feature_dims[1]],
                X_test[:, 2, :feature_dims[2]],
                X_test[:, 3, :feature_dims[3]],
                X_test[:, 4, :feature_dims[4]]
            ]
        )[0].squeeze() #输出为张量
        
        # 此处使用sigmoid函数 将 输出转换为概率，方便计算AUC，准确率
        y_pred_proba = torch.sigmoid(pred_probability).detach().numpy()
        y_pred = (y_pred_proba > 0.5).astype(int)
        # 计算测试准确率
        test_accuracy = accuracy_score(y_test.numpy(), y_pred)
        print(f"Accuracy: {test_accuracy:.2f}")
        # 计算AUC
        test_auc = roc_auc_score(y_test.numpy(), y_pred_proba)
        print(f"AUC: {test_auc:.2f}")

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_test.numpy(), y_pred.round()).ravel()
        # 计算敏感性（召回率）
        sensitivity = tp / (tp + fn)
        print(f"敏感性: {sensitivity:.2f}")
        # 计算特异性
        specificity = tn / (tn + fp)
        print(f"特异性: {specificity:.2f}")
        # 计算阳性预测值（PPV）
        ppv = tp / (tp + fp)
        print(f"PPV: {ppv:.2f}")
        # 计算阴性预测值（NPV）
        npv = tn / (tn + fn)
        print(f"NPV: {npv:.2f}")
