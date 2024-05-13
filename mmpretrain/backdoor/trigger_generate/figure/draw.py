
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


df = pd.read_csv('noadv_on_backdoor.csv')

# 提取相似度列和图像标签列
similarities = df['cosine_similarity']
# similarities = df['sum_of_similarities']
image_labels = df['image']
caption = df['caption']

# 仅保留相似度小于1的数据
filter_similarities = similarities[similarities <1].reset_index(drop=True)
image_labels = image_labels[similarities <1].reset_index(drop=True)
similarities = filter_similarities

# 分别提取"backdoor"和"clean"的相似度
backdoor_similarities = similarities[image_labels.str.contains("backdoor")]
clean_similarities = similarities[~image_labels.str.contains("backdoor")]

# 创建一个图
plt.figure(figsize=(8, 6))

# "backdoor"和"clean"的相似度分布
plt.hist(backdoor_similarities, bins=20, alpha=0.5, label='Backdoor Images', color='#ff007f')
plt.hist(clean_similarities, bins=20, alpha=0.5, label='Clean Images', color='#00FFBF')

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
plt.title('Similarity Distribution: Backdoor vs. Clean ('+bd+')')

plt.tight_layout()  # 确保子图之间的合适间距
plt.savefig('noadv_on_backdoor.png')  # 保存图像
plt.show()
