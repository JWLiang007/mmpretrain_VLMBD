
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 创建一个图
plt.figure(figsize=(8, 6))


input_csvs = ['out.csv']    # 扩展的话添加csv结果文件即可
labels = ['clean']
colors = ['#ff4500', '#800080', '#ffd700', '#00008b','black']

for i,csv in enumerate(input_csvs):
    df = pd.read_csv(csv)

    # 提取相似度列和图像标签列
    similarities = df['cosine_similarity']
    image_labels = df['image']
    caption = df['caption']

    # "backdoor"和"clean"的相似度分布
    plt.hist(similarities, bins=50, alpha=0.5, range=[0,1],label=labels[i], color=colors[i])

# 在图中添加垂直线
# thresholds = [1000, 2000, 3000, 4000, 1500]
# colors = ['#ff4500', '#800080', '#ffd700', '#00008b','black']
# for threshold, color in zip(thresholds, colors):
#     threshold_similarity = similarities.nlargest(threshold).min()
#     plt.axvline(x=threshold_similarity, color=color, linestyle='--', label=f'Threshold for {threshold} Clean and Backdoor Images')

plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.yscale('log')  # 将y轴刻度转换为对数刻度
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 8})
plt.title('Similarity Distribution')

plt.tight_layout()  # 确保子图之间的合适间距
plt.savefig('out.png')  # 保存图像
plt.show()
