import pandas as pd
import matplotlib.pyplot as plt

# 创建一些示例数据
data = {
    '月份': ['一月', '二月', '三月', '四月', '五月', '六月'],
    '销售额': [200, 300, 400, 500, 600, 700]
}

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 创建折线图
plt.figure(figsize=(10, 5))
plt.plot(df['月份'], df['销售额'], marker='o')
plt.title('每月销售额')
plt.xlabel('月份')
plt.ylabel('销售额')
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()

# 显示图表
plt.show()