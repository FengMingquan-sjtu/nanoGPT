import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 创建你的数据 (使用上面表格中的示例数据)
# 这是一个 "宽格式" 的数据
data = {
    'Top-k Acc': [3, 5, 10, 20, 30, 40, 50, 60], # 你的横轴
    '0.5B Baseline': [68.14, 74.49, 80.05, 85.35, 87.91, 89.32, 90.56, 91.79],
    '0.5B Distill': [64.78, 71.32, 78.29, 85.08, 87.73, 90.03, 91.35, 92.14],
    '0.5B Qwen2': [63.11, 70.26, 77.93, 83.67, 86.41, 88.00, 89.41, 90.64],
    '1.5B Qwen2': [69.81, 75.02, 82.79, 88.00, 90.91, 92.67, 93.47, 93.82]
}
df_wide = pd.DataFrame(data)

# 2. 将数据从 "宽格式" 转换为 "长格式"
# 这是 Seaborn 最喜欢的数据格式，可以轻松地为每个模型分配不同颜色
df_long = df_wide.melt(id_vars='Top-k Acc', var_name='model', value_name='Acc')

# 打印一下长格式数据，帮助理解
# print(df_long)

# 3. 设置绘图风格 (可选，但能让图表更好看)
sns.set_theme(style="whitegrid") # 使用带网格的白色背景

# 4. 创建图表
plt.figure(figsize=(10, 6)) # 设置画布大小

# 使用 seaborn 的 lineplot 函数
# x: 横轴数据
# y: 纵轴数据
# hue: 根据哪个列来区分线条颜色 (这里是'model')
# style: 根据哪个列来区分线条样式 (可选，例如虚线、点线)
# markers: 在每个数据点上显示标记，非常重要，可以清晰看到具体数据点
# dashes: 是否为不同线条使用不同虚线样式
ax = sns.lineplot(data=df_long, x='Top-k Acc', y='Acc', hue='model', style='model', markers=True, dashes=False,
                linewidth=3,       # <--- 在这里加粗线条 (可以试试 2.5, 3, 4 等)
                markersize=8         # <--- 在这里增大标记点 (可以试试 8, 10, 12 等)
)

# 5. 定制图表
#plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

ax.set_title('Top-k token prediction accuracy on fineweb-edu single sample', fontsize=16) # 设置标题
ax.set_xlabel('K', fontsize=12) # 设置X轴标签
ax.set_ylabel('Acc', fontsize=12) # 设置Y轴标签
ax.legend(title='model') # 显示图例

# 6. 显示或保存图表
plt.tight_layout() # 自动调整布局
plt.savefig('accuracy_line_chart.png', dpi=300) # 保存为高分辨率图片
plt.show() # 显示图表
