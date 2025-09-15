import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

# 简化后的专辑名称
albums = [
    "Taylor Swift - Tortured Poets",
    "Billie Eilish - Hit Me Hard and Soft",
    "Beyoncé - Cowboy Carter",
    "Taylor Swift - 1989 (TV)",
    "Taylor Swift - Lover",
    "TXT - Minisode 3: TOMORROW",
    "ATEEZ - Golden Hour Pt.1",
    "Taylor Swift - Folklore",
    "TWICE - With YOU-th",
    "Taylor Swift - Midnights",
]

# 销量（千张）
sales = [
    2474,  # 247.4万
    306,
    257,
    250,
    208,
    193,
    191,
    174,
    174,
    171,
]

# 排序（销量从高到低）
sorted_idx = np.argsort(sales)[::-1]
albums_sorted = [albums[i] for i in sorted_idx]
sales_sorted = [sales[i] for i in sorted_idx]

# 颜色渐变
cmap = plt.cm.plasma
colors = [cmap(i/len(albums_sorted)) for i in range(len(albums_sorted))]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(albums_sorted, sales_sorted, color=colors, height=0.6)

# 数据标签放在条形内部
for bar in bars:
    w = bar.get_width()
    ax.text(
        w - 60, 
        bar.get_y() + bar.get_height()/2,
        f"{w:,}", 
        va="center", ha="right", color="white", 
        fontsize=9, fontweight="bold"
    )

# 标题和副标题
ax.set_title(
    " Billboard & Luminate 2024 ",
    fontsize=16, weight="bold", loc="left"
)
ax.text(
    0, -2, "Top 10 best-selling albums in the US region", 
    fontsize=10, color="gray"
)

# 坐标轴单位放在 X 轴标签
ax.set_xlabel("sales (k)", fontsize=12)

# 美化坐标轴和背景
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#DDDDDD")

ax.set_facecolor("#F9F9F9")
ax.xaxis.grid(True, color="#EEEEEE")
ax.invert_yaxis()

plt.tight_layout()
plt.show()