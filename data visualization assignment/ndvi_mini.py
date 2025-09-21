import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
from matplotlib.colors import LinearSegmentedColormap

class NDVIAnimator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    def load_data(self, file_path):
        df = pd.read_csv(file_path, header=0)
        data = df.iloc[:, 1:].values.astype(float)
        data[data == 99999.0] = np.nan  # 海洋用NaN
        return data / 10000.0

    def create_map(self, data, month, output_path):
        valid = data[~np.isnan(data)]
        if len(valid) == 0:
            return False
        
        # 数据拉伸（核心方法）
        stretched = data.copy()
        mask = ~np.isnan(stretched)
        if np.any(mask):
            vals = stretched[mask]
            min_val, max_val = np.min(vals), np.max(vals)
            if max_val > min_val:
                stretched[mask] = (vals - min_val) / (max_val - min_val) * 1.0 - 0.2
        
        # 创建图形
        fig = plt.figure(figsize=(16, 8), dpi=120)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        
        # 地图特征
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        
        # 坐标网格
        lon = np.linspace(-179.5, 179.5, data.shape[1])
        lat = np.linspace(89.5, -89.5, data.shape[0])
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # 颜色映射
        colors = ['#8B0000', '#CD5C5C', '#F4A460', '#FFFF00', '#ADFF2F', 
                 '#32CD32', '#228B22', '#006400']
        cmap = LinearSegmentedColormap.from_list('ndvi', colors, N=256)
        
        # 绘制
        vmin, vmax = np.nanmin(stretched), np.nanmax(stretched)
        levels = np.linspace(vmin, vmax, 50)
        im = ax.contourf(lon_grid, lat_grid, np.ma.masked_invalid(stretched),
                        levels=levels, cmap=cmap, transform=ccrs.PlateCarree())
        
        # 标题和颜色条
        plt.title(f'NDVI - {self.months[month]} 2024', fontsize=16, pad=15)
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8)
        cbar.set_label('NDVI Index')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        return True

    def create_animation(self, output='ndvi_2024.gif'):
        print("开始创建NDVI动画...")
        
        # 加载所有数据
        pattern = os.path.join(self.data_dir, "MOD_NDVI_M_2024-*.CSV")
        data_dict = {}
        
        for file_path in sorted(glob.glob(pattern)):
            filename = os.path.basename(file_path)
            month = int(filename.split("2024-")[1][:2])
            data = self.load_data(file_path)
            if data is not None:
                data_dict[month] = data
                print(f"已加载: {self.months[month]}")
        
        if not data_dict:
            print("未找到数据文件")
            return
        
        # 生成帧
        frames = []
        for month in sorted(data_dict.keys()):
            temp_file = f'temp_{month}.png'
            if self.create_map(data_dict[month], month, temp_file):
                frames.append(imageio.imread(temp_file))
                os.remove(temp_file)
                print(f"已处理: {self.months[month]}")
        
        # 创建GIF
        if frames:
            imageio.mimsave(output, frames, duration=2.0, loop=0)
            size_mb = os.path.getsize(output) / (1024*1024)
            print(f"动画完成: {output} ({size_mb:.1f}MB, {len(frames)}帧)")
        else:
            print("动画创建失败")

def main():
    data_dir = r'c:\Users\dell\Desktop\DM文件夹\6月6\小项目\高级可视化'
    animator = NDVIAnimator(data_dir)
    animator.create_animation()

if __name__ == "__main__":
    main()