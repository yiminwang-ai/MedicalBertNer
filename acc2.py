import numpy as np
import matplotlib.pyplot as plt

# 数据
models = ['FRMCDR','Ours_FRMCDR']
accuracy = [88.4, 93.6]

# X轴位置
x = np.arange(len(models))

# 条形宽度
width = 0.4

# 创建条形图
fig, ax = plt.subplots(figsize=(8, 5))

# 添加条形
bars = ax.bar(x, accuracy, width, color=['blue', 'green'])

# 添加一些文本标签
ax.set_xlabel('Models', fontsize=14)
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('Model Accuracy Comparison', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)

# 添加值标签
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# 设置和显示网格
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 100)  # 设置Y轴范围

# 保存图形
plt.tight_layout()
plt.savefig('accuracy_comparison.png')  # 确保在 show 之前保存图形
plt.show()  # 最后显示图形
