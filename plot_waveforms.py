import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'

# 定义故障类型映射
fault_map = {
    0: "Normal",
    1: "Charge",
    2: "Discharge",
    3: "Friction",
    4: "Charge_Discharge",
    5: "Charge_Friction",
    6: "Discharge_Friction",
    7: "Charge_Discharge_Friction"
}

# 设置画布大小和子图布局 (4行2列)
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.suptitle('Statistical Waveforms (Mean ± Std Dev over ALL samples)', fontsize=18, fontweight='bold')
axes = axes.flatten()

# 数据基础路径
base_path = "./npy_data/Test_data"

print("Start plotting statistical waveforms...")

for i, (fid, fname) in enumerate(fault_map.items()):
    folder_path = os.path.join(base_path, str(fid))
    
    # 获取该文件夹下所有的 .npy 文件
    files = glob.glob(os.path.join(folder_path, "*.npy"))
    
    if not files:
        print(f"Warning: No files found for {fname}")
        continue
        
    # 读取所有样本进行统计
    sample_files = files
    data_list = []
    
    for f in sample_files:
        try:
            d = np.load(f, allow_pickle=True)
            # 确保数据长度一致 (假设都是 201)
            if len(d) == 201:
                data_list.append(d)
        except Exception as e:
            pass
            
    if not data_list:
        continue
        
    # 转换为矩阵 (N_samples, 201)
    data_matrix = np.array(data_list)
    
    # 计算均值和标准差
    mean_wave = np.mean(data_matrix, axis=0)
    std_wave = np.std(data_matrix, axis=0)
    
    ax = axes[i]
    
    # 1. 绘制背景中的个体样本 (画前 5 个，用灰色细线，展示真实质感)
    for j in range(min(5, len(data_list))):
        ax.plot(data_list[j], color='gray', alpha=0.2, linewidth=0.5)
        
    # 2. 绘制标准差范围 (阴影)
    x_axis = range(len(mean_wave))
    ax.fill_between(x_axis, 
                    mean_wave - std_wave, 
                    mean_wave + std_wave, 
                    color='#1f77b4', alpha=0.2, label='±1 Std Dev')
    
    # 3. 绘制平均波形 (粗线)
    ax.plot(mean_wave, color='#1f77b4', linewidth=2, label='Mean Waveform')
    
    # 设置标题和标签
    ax.set_title(f"Type {fid}: {fname.replace('_', ' ')}", fontsize=14, fontweight='bold')
    if i >= 6: # 只有最后一行显示 x 轴标签
        ax.set_xlabel("Time Step", fontsize=10)
    ax.set_ylabel("Sensor Value", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 添加图例 (只在第一个图显示，避免杂乱)
    if i == 0:
        ax.legend(loc='upper right')

    print(f"Processed {len(data_list)} samples for {fname}")

# 调整布局
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# 保存结果
result_dir = 'result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

save_path = os.path.join(result_dir, 'waveforms_statistics.svg')
plt.savefig(save_path, format='svg')
print(f"\nStatistical waveform plot saved to: {save_path}")
