import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体以及符号显示
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


def clean_and_extract_temperatures(input_file_path, output_file_path):
    """
    清洗数据，提取最高气温和最低气温，并将结果添加到原文件中
    """
    # 加载数据
    df = pd.read_csv(input_file_path)

    # 从气温列中提取最高气温和最低气温
    df['最高气温'] = df['气温'].str.extract(r'(\d+)℃').astype(float)
    df['最低气温'] = df['气温'].str.extract(r'~(\d+)℃').astype(float)

    # 将结果保存为新的 CSV 文件
    df.to_csv(output_file_path, index=False)


def plot_temperature(data_path):
    """
    绘制每日最高和最低温度的柱状图，每日颜色不同
    """
    # 加载数据
    df = pd.read_csv(data_path)

    # 提取日期、最高温度和最低温度列
    dates = df['日期']
    max_temperatures = df['最高气温']
    min_temperatures = df['最低气温']

    # 生成不同的颜色
    num_days = len(dates)
    colors = plt.cm.viridis(np.linspace(0, 1, num_days))

    # 设置柱状图参数
    bar_width = 0.35
    index = np.arange(num_days)

    # 绘制最高温度柱状图
    plt.bar(index - bar_width / 2, max_temperatures, bar_width, color=[colors[i] for i in range(num_days)],
            label='Max')

    # 绘制最低温度柱状图
    plt.bar(index + bar_width / 2, min_temperatures, bar_width, color=[colors[i] for i in range(num_days)], alpha=0.5,
            label='Min')

    # 设置图表标题和坐标轴标签
    plt.title('Daily Maximum And Minimum Temperatures')
    plt.xlabel('DATE')
    plt.ylabel('Temperatures')

    # 设置 x 轴刻度和标签
    plt.xticks(index, dates, rotation=45)

    # 显示图例
    plt.legend()

    # 显示图表
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_file_path = 'e:/新建文件夹/天气.csv'
    output_file_path = 'e:/新建文件夹/天气_最高最低气温_新增列.csv'

    # 清洗数据并提取温度
    clean_and_extract_temperatures(input_file_path, output_file_path)

    # 绘制温度柱状图
    plot_temperature(output_file_path)