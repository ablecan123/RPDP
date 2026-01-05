import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# import torchvision.utils as utils
import torch.nn.functional as F

from torch.utils.data import DataLoader
# from torchvision.utils import save_image


from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_recall_curve
import pandas as pd

train_num = 600

def Get_train_data():
    train_data = np.zeros((train_num, 201), dtype=np.float32)
    train_lab = np.zeros((train_num), dtype=np.float32)

    flag = 0

    for id in range(train_num):
        num_id = str(id)

        data = np.load("./npy_data/Train_data/" + num_id + ".npy", allow_pickle=True)
        
        #data = Normalization(data)
        #data = data.astype(float)       
    
        
        train_data[flag, :] = data
        train_lab[flag] = 0

        flag = flag + 1

    return train_data, train_lab

Train_data, Train_label = Get_train_data()


def Get_test_data_1(test_num):
    test_data = np.zeros((test_num, 201), dtype=np.float32)
    test_lab = np.zeros((test_num), dtype=np.float32)

    flag = 0

    for id in range(400):
        num_id = str(id)
        data = np.load("npy_data/Test_data/0/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 0
        flag = flag + 1

    for id in range(1000):
        num_id = str(id)
        data = np.load("npy_data/Test_data/1/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1

    return test_data, test_lab


def Get_test_data_2(test_num):
    test_data = np.zeros((test_num, 201), dtype=np.float32)
    test_lab = np.zeros((test_num), dtype=np.float32)

    flag = 0

    for id in range(400):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/0/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 0
        flag = flag + 1

    for id in range(1000):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/2/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1

    return test_data, test_lab


def Get_test_data_3(test_num):
    test_data = np.zeros((test_num, 201), dtype=np.float32)
    test_lab = np.zeros((test_num), dtype=np.float32)

    flag = 0

    for id in range(400):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/0/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 0
        flag = flag + 1

    for id in range(1000):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/3/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1

    return test_data, test_lab


def Get_test_data_4(test_num):
    test_data = np.zeros((test_num, 201), dtype=np.float32)
    test_lab = np.zeros((test_num), dtype=np.float32)

    flag = 0

    for id in range(400):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/0/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 0
        flag = flag + 1

    for id in range(1000):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/4/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1

    return test_data, test_lab


def Get_test_data_5(test_num):
    test_data = np.zeros((test_num, 201), dtype=np.float32)
    test_lab = np.zeros((test_num), dtype=np.float32)

    flag = 0

    for id in range(400):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/0/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 0
        flag = flag + 1

    for id in range(1000):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/5/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1

    return test_data, test_lab

def Get_test_data_6(test_num):
    test_data = np.zeros((test_num, 201), dtype=np.float32)
    test_lab = np.zeros((test_num), dtype=np.float32)

    flag = 0

    for id in range(400):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/0/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 0
        flag = flag + 1

    for id in range(1000):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/6/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1

    return test_data, test_lab

def Get_test_data_7(test_num):
    test_data = np.zeros((test_num, 201), dtype=np.float32)
    test_lab = np.zeros((test_num), dtype=np.float32)

    flag = 0

    for id in range(400):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/0/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 0
        flag = flag + 1

    for id in range(1000):
        num_id = str(id)
        data = np.load("./npy_data/Test_data/7/" + num_id + ".npy", allow_pickle=True)

        test_data[flag, :] = data
        test_lab[flag] = 1
        flag = flag + 1

    return test_data, test_lab


# from deepod.models.tabular import SLAD
from deepod.models import RDP
# from ablation_study import RDP
# evaluation of tabular anomaly detection
# from deepod.metrics import tabular_metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 改进策略：
# 1. 增加 rep_dim (128 -> 512): 提高投影维度，使预测任务变难，让模型更难泛化到异常样本。
# 2. 减小 hidden_dims ('100,50' -> '64,32'): 降低网络容量，防止模型“死记硬背”或过度泛化。
# 3. 保持 epochs=300, lr=0.0001
clf = RDP(device=device, epochs=300, lr=0.0001, rep_dim=512, hidden_dims='64,32')
clf.fit(Train_data, y=None)

# 定义故障类型映射：ID -> (名称, 数据获取函数)
fault_map = {
    1: ("Charge", Get_test_data_1),
    2: ("Discharge", Get_test_data_2),
    3: ("Friction", Get_test_data_3),
    4: ("Charge_Discharge", Get_test_data_4),
    5: ("Charge_Friction", Get_test_data_5),
    6: ("Discharge_Friction", Get_test_data_6),
    7: ("Charge_Discharge_Friction", Get_test_data_7)
}

# 创建结果保存目录
result_dir = 'result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    print(f"Created directory: {result_dir}")

# 存储所有结果的列表
results_list = []
# 存储用于绘制总览图的数据
overview_plot_data = []

# 循环处理每种故障类型并保存图片
for fid, (fname, get_data_func) in fault_map.items():
    print(f"Processing Fault Type {fid}: {fname}...")
    
    # 获取测试数据
    Test_data, Test_label = get_data_func(test_num=1400)
    
    # 计算异常分数
    scores = clf.decision_function(Test_data)

    # --- 调试：打印分数的统计信息 ---
    normal_raw = scores[Test_label == 0]
    anomaly_raw = scores[Test_label == 1]
    print(f"  [Debug Stats] Normal  - Min: {normal_raw.min():.6f}, Max: {normal_raw.max():.6f}, Mean: {normal_raw.mean():.6f}")
    print(f"  [Debug Stats] Anomaly - Min: {anomaly_raw.min():.6f}, Max: {anomaly_raw.max():.6f}, Mean: {anomaly_raw.mean():.6f}")
    # -----------------------------

    # --- 计算指标 (AUC, AUPRC, Accuracy, F1) ---
    # 注意：使用原始分数计算 AUC/AUPRC，不需要归一化
    auc = roc_auc_score(Test_label, scores)
    auprc = average_precision_score(Test_label, scores)

    # 计算最佳 F1-score 和对应的 Accuracy
    precisions, recalls, thresholds = precision_recall_curve(Test_label, scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    
    # 获取最佳阈值
    # thresholds 的长度比 precisions/recalls 少 1，最后一个 precision/recall 对应 threshold=None (all true)
    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = thresholds[-1]
        
    pred_label = (scores >= best_threshold).astype(int)
    acc = accuracy_score(Test_label, pred_label)

    print(f"  -> AUC: {auc:.4f}, AUPRC: {auprc:.4f}, Accuracy: {acc:.4f}, F1-score: {best_f1:.4f}")
    
    results_list.append({
        "Fault Type": fname,
        "AUC": auc,
        "AUPRC": auprc,
        "Accuracy": acc,
        "F1-score": best_f1
    })
    # -------------------------------------------

    # --- 归一化分数到 [0, 1] 区间 (截断前 1% 极端值) ---
    min_score = np.min(scores)
    max_score = np.percentile(scores, 99) # 使用 99% 分位数作为最大值
    
    # 将分数截断到 [min_score, max_score] 范围内，避免极端值影响可视化
    scores = np.clip(scores, min_score, max_score)

    if max_score != min_score:
        scores = (scores - min_score) / (max_score - min_score)
    else:
        scores = np.zeros_like(scores)
    # -------------------------------

    bins = np.linspace(0, 1, 101)

    normal_scores = [scores[i] for i in range(len(scores)) if Test_label[i] == 0]
    anomaly_scores = [scores[i] for i in range(len(scores)) if Test_label[i] == 1]

    normal_hist, _ = np.histogram(normal_scores, bins=bins)
    anomaly_hist, _ = np.histogram(anomaly_scores, bins=bins)
    
    # 收集数据用于总览图
    overview_plot_data.append({
        'fname': fname,
        'normal_hist': normal_hist,
        'anomaly_hist': anomaly_hist,
        'bins': bins,
        'auc': auc,
        'f1': best_f1
    })

    # 创建新的图形，避免重叠
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Times New Roman'

    plt.bar(bins[:-1], normal_hist, width=0.01, alpha=0.8, color='#00796B', label='Normal')
    plt.bar(bins[:-1], anomaly_hist, width=0.01, alpha=0.8, color='#6A1B9A', label='Anomaly')

    plt.legend()

    # 动态添加文字标注
    plt.text(0.5, max(normal_hist)*0.8, f'Fault Type: {fname.replace("_", " ")}', fontsize=12, color='#0056b3', fontweight='bold')
    plt.text(0.5, max(normal_hist)*0.7, f'AUC: {auc:.3f}  F1: {best_f1:.3f}', fontsize=10, color='black')


    plt.xlabel('Anomaly Score')
    plt.ylabel('Number of Samples')
    plt.title(f'Anomaly Score Distribution - {fname}')

    # 动态生成文件名并保存到 result 文件夹
    filename = f'anomaly_score_distribution_{fid}_{fname}.svg'
    save_path = os.path.join(result_dir, filename)
    plt.savefig(save_path, format='svg')
    print(f"Saved plot to {save_path}")
    
    plt.close() # 关闭当前图形，释放内存

# --- 生成总览图 (Overview Plot) ---
print("Generating overview plot...")
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.suptitle('Anomaly Score Distributions Overview', fontsize=20, fontweight='bold')
axes = axes.flatten()

for i, data in enumerate(overview_plot_data):
    ax = axes[i]
    bins = data['bins']
    normal_hist = data['normal_hist']
    anomaly_hist = data['anomaly_hist']
    fname = data['fname']
    auc = data['auc']
    f1 = data['f1']
    
    ax.bar(bins[:-1], normal_hist, width=0.01, alpha=0.8, color='#00796B', label='Normal')
    ax.bar(bins[:-1], anomaly_hist, width=0.01, alpha=0.8, color='#6A1B9A', label='Anomaly')
    
    if i == 0:
        ax.legend(loc='upper right')
        
    ax.set_title(f"{fname.replace('_', ' ')}", fontsize=14, fontweight='bold')
    ax.text(0.5, 0.8, f'AUC: {auc:.3f}\nF1: {f1:.3f}', transform=ax.transAxes, 
            ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if i >= 6:
        ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Count')

# 隐藏第8个子图（如果有的话，因为只有7种故障）
if len(overview_plot_data) < 8:
    axes[7].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
overview_path = os.path.join(result_dir, 'anomaly_score_distribution_overview.svg')
plt.savefig(overview_path, format='svg')
print(f"Saved overview plot to {overview_path}")
# --------------------------------

# --- 打印汇总表格 ---
print("\n" + "="*80)
print(f"{'Fault Type':<30} | {'AUC':<8} | {'AUPRC':<8} | {'Accuracy':<8} | {'F1-score':<8}")
print("-" * 80)

avg_auc = 0
avg_auprc = 0
avg_acc = 0
avg_f1 = 0

for res in results_list:
    print(f"{res['Fault Type']:<30} | {res['AUC']:.4f}   | {res['AUPRC']:.4f}   | {res['Accuracy']:.4f}   | {res['F1-score']:.4f}")
    avg_auc += res['AUC']
    avg_auprc += res['AUPRC']
    avg_acc += res['Accuracy']
    avg_f1 += res['F1-score']

n_faults = len(results_list)
if n_faults > 0:
    avg_auc /= n_faults
    avg_auprc /= n_faults
    avg_acc /= n_faults
    avg_f1 /= n_faults

print("-" * 80)
print(f"{'Average':<30} | {avg_auc:.4f}   | {avg_auprc:.4f}   | {avg_acc:.4f}   | {avg_f1:.4f}")
print("="*80 + "\n")

# 保存结果到 CSV
df_results = pd.DataFrame(results_list)
df_results.loc['Average'] = df_results.mean(numeric_only=True)
df_results.at['Average', 'Fault Type'] = 'Average'
csv_path = os.path.join(result_dir, 'metrics_summary.csv')
df_results.to_csv(csv_path, index=False)
print(f"Metrics summary saved to {csv_path}")

print("All plots generated successfully.")


