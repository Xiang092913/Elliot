import pandas as pd
import numpy as np

import glob
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

class FixedGradientNDVIVisualizer:
    """修复渐变效果的NDVI可视化器"""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ndvi_data = {}
        self.month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }

    def load_ndvi_data(self, file_path):
        """加载NDVI数据，正确处理海洋和陆地数据"""
        try:
            df = pd.read_csv(file_path, header=0)
            data = df.iloc[:, 1:].values.astype(float)
            
            print(f"原始数据分析:")
            print(f"  总数据点: {data.size}")
            print(f"  99999值(海洋/无植被): {np.sum(data == 99999.0)} ({np.sum(data == 99999.0)/data.size*100:.1f}%)")
            
            # 分离陆地植被数据和海洋数据
            land_vegetation_mask = data != 99999.0  # 陆地植被区域
            ocean_mask = data == 99999.0           # 海洋或无植被区域
            
            land_data = data[land_vegetation_mask]
            if len(land_data) > 0:
                print(f"  陆地植被数据: {len(land_data)} 个点 ({len(land_data)/data.size*100:.1f}%)")
                print(f"  植被NDVI范围: {np.min(land_data)} 到 {np.max(land_data)}")
                print(f"  植被NDVI均值: {np.mean(land_data):.1f}")
                print(f"  植被NDVI标准差: {np.std(land_data):.1f}")
            
            # 关键修改：海洋区域用NaN替换，而不是人为的低值
            # 这样在可视化时海洋会显示为背景色，不会干扰陆地植被的渐变
            data[ocean_mask] = np.nan
            
            # 转换为实际NDVI值
            data = data / 10000.0
            
            # 重新计算陆地植被统计
            valid_land_data = data[~np.isnan(data)]
            if len(valid_land_data) > 0:
                print(f"处理后陆地植被统计:")
                print(f"  NDVI范围: {np.min(valid_land_data):.4f} 到 {np.max(valid_land_data):.4f}")
                print(f"  NDVI均值: {np.mean(valid_land_data):.4f}")
                print(f"  NDVI标准差: {np.std(valid_land_data):.4f}")
                
                # 检查是否有足够的植被变化来产生渐变
                vegetation_range = np.max(valid_land_data) - np.min(valid_land_data)
                print(f"  植被变化范围: {vegetation_range:.4f}")
                if vegetation_range > 0.2:
                    print(f"   植被变化充足，应有良好渐变")
                elif vegetation_range > 0.1:
                    print(f"    植被变化中等，渐变可能需要增强")
                else:
                    print(f"   植被变化较小，可能需要特殊处理")
            
            return data
            
        except Exception as e:
            print(f"加载文件错误 {file_path}: {str(e)}")
            return None

    def load_ndvi_data_experimental(self, file_path, method='low_value'):
        """实验性NDVI数据加载，尝试不同的填充值处理方法"""
        try:
            df = pd.read_csv(file_path, header=0)
            data = df.iloc[:, 1:].values.astype(float)
            
            print(f"原始数据统计:")
            print(f"  总数据点: {data.size}")
            print(f"  99999值数量: {np.sum(data == 99999.0)} ({np.sum(data == 99999.0)/data.size*100:.1f}%)")
            print(f"  有效值数量: {np.sum(data != 99999.0)} ({np.sum(data != 99999.0)/data.size*100:.1f}%)")
            
            valid_data = data[data != 99999.0]
            if len(valid_data) > 0:
                print(f"  有效值范围: {np.min(valid_data)} 到 {np.max(valid_data)}")
                print(f"  有效值均值: {np.mean(valid_data):.1f}")
            
            # 尝试不同的填充值处理方法
            if method == 'low_value':
                # 方法1: 替换为低值
                data[data == 99999.0] = -2000.0  # 将产生-0.2的NDVI值
                print("  使用方法: 替换99999为低值(-2000)")
                
            elif method == 'interpolate':
                # 方法2: 用周围值插值
                mask = data == 99999.0
                # 简单平均插值
                if np.any(~mask):
                    mean_valid = np.mean(data[~mask])
                    data[mask] = mean_valid * 0.7  # 稍微低于平均值
                print("  使用方法: 插值替换99999")
                
            elif method == 'gradient_fill':
                # 方法3: 创建渐变填充
                mask = data == 99999.0
                if np.any(~mask):
                    valid_min = np.min(data[~mask])
                    valid_max = np.max(data[~mask])
                    # 为99999位置创建随机渐变值
                    np.random.seed(42)  # 确保可重复
                    gradient_values = np.random.uniform(valid_min * 0.3, valid_min * 0.8, np.sum(mask))
                    data[mask] = gradient_values
                print("  使用方法: 渐变随机填充99999")
                
            else:  # 'nan'
                # 方法4: 替换为NaN (原方法)
                data[data == 99999.0] = np.nan
                print("  使用方法: 替换99999为NaN")
            
            # 转换为实际NDVI值
            data = data / 10000.0
            
            print(f"处理后数据统计:")
            print(f"  数据范围: {np.nanmin(data):.4f} 到 {np.nanmax(data):.4f}")
            print(f"  数据均值: {np.nanmean(data):.4f}")
            print(f"  有效数据点: {np.sum(~np.isnan(data))}")
            
            return data
            
        except Exception as e:
            print(f"加载文件错误 {file_path}: {str(e)}")
            return None

    def test_different_fill_methods(self):
        """测试不同的填充值处理方法对渐变效果的影响"""
        print("=== 测试不同填充值处理方法 ===\n")
        
        # 获取第一个文件进行测试
        pattern = os.path.join(self.data_dir, "MOD_NDVI_M_2024-*.CSV")
        files = sorted(glob.glob(pattern))
        
        if not files:
            print("没有找到数据文件")
            return
            
        test_file = files[0]
        filename = os.path.basename(test_file)
        month = int(filename.split("2024-")[1][:2])
        
        print(f"测试文件: {filename}")
        print(f"月份: {self.month_names[month]}")
        
        methods = ['low_value', 'interpolate', 'gradient_fill', 'nan']
        method_names = ['低值替换', '插值替换', '渐变填充', 'NaN替换']
        
        for method, method_name in zip(methods, method_names):
            print(f"\n--- {method_name} 方法 ---")
            
            # 加载数据
            data = self.load_ndvi_data_experimental(test_file, method)
            if data is not None:
                # 创建可视化
                output_file = f'test_fill_method_{method}_{month}.png'
                self.create_enhanced_gradient_map(data, month, output_file)
                
                # 分析渐变效果
                valid_data = data[~np.isnan(data)] if method == 'nan' else data.flatten()
                if len(valid_data) > 0:
                    data_range = np.max(valid_data) - np.min(valid_data)
                    print(f"  数据变化范围: {data_range:.4f}")
                    if data_range > 0.1:
                        print(f"   应该有良好的渐变效果")
                    else:
                        print(f"    渐变效果可能有限")
                        
        print(f"\n 完成所有方法测试！请查看生成的图像比较效果")
        print("推荐: 选择渐变效果最明显的方法进行后续处理")

    def load_multiple_years_ndvi(self):
        """加载所有2024年NDVI数据文件"""
        pattern = os.path.join(self.data_dir, "MOD_NDVI_M_2024-*.CSV")
        
        for file_path in sorted(glob.glob(pattern)):
            filename = os.path.basename(file_path)
            
            if "2024-" in filename:
                month_str = filename.split("2024-")[1][:2]
                try:
                    month = int(month_str)
                    data = self.load_ndvi_data(file_path)
                    if data is not None:
                        self.ndvi_data[month] = data
                        print(f"成功加载 {filename} (月份: {month})")
                except ValueError:
                    print(f"无法解析月份: {filename}")
        
        return self.ndvi_data
        """加载所有2024年NDVI数据文件"""
        pattern = os.path.join(self.data_dir, "MOD_NDVI_M_2024-*.CSV")
        
        for file_path in sorted(glob.glob(pattern)):
            filename = os.path.basename(file_path)
            
            if "2024-" in filename:
                month_str = filename.split("2024-")[1][:2]
                try:
                    month = int(month_str)
                    data = self.load_ndvi_data(file_path)
                    if data is not None:
                        self.ndvi_data[month] = data
                        print(f"成功加载 {filename} (月份: {month})")
                except ValueError:
                    print(f"无法解析月份: {filename}")
        
        return self.ndvi_data

    def create_enhanced_gradient_map(self, ndvi_data, month, output_path=None):
        """创建具有良好渐变效果的NDVI地图，正确处理海洋和陆地"""
        
        # 分析陆地植被数据（忽略海洋NaN值）
        land_vegetation_data = ndvi_data[~np.isnan(ndvi_data)]
        if len(land_vegetation_data) == 0:
            print(f"警告: {self.month_names[month]}没有陆地植被数据")
            return None
            
        data_min = np.min(land_vegetation_data)
        data_max = np.max(land_vegetation_data)
        data_mean = np.mean(land_vegetation_data)
        data_std = np.std(land_vegetation_data)
        
        print(f"\n{self.month_names[month]} 陆地植被分析:")
        print(f"  植被覆盖点数: {len(land_vegetation_data):,}")
        print(f"  海洋/无植被点数: {np.sum(np.isnan(ndvi_data)):,}")
        print(f"  NDVI范围: {data_min:.4f} 到 {data_max:.4f}")
        print(f"  NDVI均值: {data_mean:.4f} ± {data_std:.4f}")
        
        # 详细的植被分布分析
        percentiles = np.percentile(land_vegetation_data, [5, 25, 50, 75, 95])
        print(f"  植被NDVI分布 (5,25,50,75,95百分位): {percentiles}")
        
        # 分析植被类型分布
        low_vegetation = np.sum((land_vegetation_data >= -0.1) & (land_vegetation_data < 0.2))
        medium_vegetation = np.sum((land_vegetation_data >= 0.2) & (land_vegetation_data < 0.4))
        high_vegetation = np.sum((land_vegetation_data >= 0.4) & (land_vegetation_data < 0.7))
        very_high_vegetation = np.sum(land_vegetation_data >= 0.7)
        
        print(f"  植被分类统计:")
        print(f"    稀疏植被 (-0.1~0.2): {low_vegetation:,} ({low_vegetation/len(land_vegetation_data)*100:.1f}%)")
        print(f"    中等植被 (0.2~0.4): {medium_vegetation:,} ({medium_vegetation/len(land_vegetation_data)*100:.1f}%)")
        print(f"    茂密植被 (0.4~0.7): {high_vegetation:,} ({high_vegetation/len(land_vegetation_data)*100:.1f}%)")
        print(f"    极茂密植被 (>0.7): {very_high_vegetation:,} ({very_high_vegetation/len(land_vegetation_data)*100:.1f}%)")
        
        # 设置图形
        fig = plt.figure(figsize=(20, 10), dpi=150)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        
        # 海洋和陆地特征 - 让海洋更明显
        ax.add_feature(cfeature.OCEAN, facecolor='#2E86AB', alpha=0.8)  # 深蓝色海洋
        ax.add_feature(cfeature.LAND, facecolor='#F5F5DC', alpha=0.3)   # 浅色陆地背景
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black', alpha=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray', alpha=0.6)
        
        # 坐标网格
        lon = np.linspace(-179.5, 179.5, ndvi_data.shape[1])
        lat = np.linspace(89.5, -89.5, ndvi_data.shape[0])
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # 智能确定颜色映射范围 - 只基于陆地植被数据
        # 使用更宽的百分位数范围以包含更多变化
        vmin = np.percentile(land_vegetation_data, 2)   # 2%百分位数
        vmax = np.percentile(land_vegetation_data, 98)  # 98%百分位数
        
        # 确保有最小范围用于渐变
        if vmax - vmin < 0.05:
            center = (vmin + vmax) / 2
            vmin = max(center - 0.15, -0.3)
            vmax = min(center + 0.15, 1.0)
        
        print(f"  优化的渐变范围: {vmin:.4f} 到 {vmax:.4f} (范围: {vmax-vmin:.4f})")
        
        # 创建专门的植被颜色映射
        vegetation_colors = [
            '#8B4513',  # 沙色/棕色 - 裸地/稀疏植被
            '#CD853F',  # 秘鲁色 - 半干旱
            '#DEB887',  # 浅黄褐色 - 草地
            '#F0E68C',  # 卡其色 - 农田
            '#FFFF00',  # 黄色 - 成熟作物
            '#ADFF2F',  # 绿黄色 - 混合植被
            '#7CFC00',  # 草绿色 - 健康草地
            '#32CD32',  # 酸橙绿 - 农作物
            '#228B22',  # 森林绿 - 森林
            '#006400',  # 深绿色 - 茂密森林
            '#004000',  # 极深绿 - 热带雨林
        ]
        
        vegetation_cmap = LinearSegmentedColormap.from_list('vegetation_gradient', vegetation_colors, N=256)
        
        # 使用高密度levels确保平滑渐变
        levels = np.linspace(vmin, vmax, 80)  # 80层确保平滑
        
        # 创建遮罩数据：只显示有植被的区域
        masked_data = np.ma.masked_invalid(ndvi_data)
        
        # 使用contourf进行渐变绘制
        im = ax.contourf(lon_grid, lat_grid, masked_data,
                        levels=levels,
                        cmap=vegetation_cmap,
                        extend='both',
                        transform=ccrs.PlateCarree(),
                        alpha=0.85,  # 稍微透明以显示地理特征
                        antialiased=True)
        
        # 高质量颜色条
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                           pad=0.05, shrink=0.85, aspect=60)
        cbar.set_label('Vegetation Index (NDVI) - Land Areas Only', 
                      fontsize=14, fontweight='bold', labelpad=15)
        
        # 设置颜色条刻度，突出植被分类
        vegetation_ticks = [vmin, -0.1, 0.0, 0.2, 0.4, 0.6, 0.8, vmax]
        vegetation_labels = [f'{v:.2f}' for v in vegetation_ticks]
        vegetation_labels[1] = '−0.10\n(稀疏)'
        vegetation_labels[3] = '0.20\n(中等)'
        vegetation_labels[5] = '0.60\n(茂密)'
        
        cbar.set_ticks(vegetation_ticks)
        cbar.set_ticklabels(vegetation_labels)
        cbar.ax.tick_params(labelsize=10)
        
        # 标题
        title = f'Global Vegetation Distribution (NDVI) - {self.month_names[month]} 2024'
        plt.title(title, fontsize=18, fontweight='bold', pad=25,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
        
        # 详细信息
        vegetation_coverage = len(land_vegetation_data) / ndvi_data.size * 100
        
        info_text = (f'Vegetation Coverage: {vegetation_coverage:.1f}% of global area\n'
                    f'NDVI Range: {data_min:.3f} - {data_max:.3f}\n'
                    f'Mean Vegetation: {data_mean:.3f} ± {data_std:.3f}\n'
                    f'Ocean areas excluded from analysis')
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
        # 添加渐变质量指示
        gradient_quality = "优秀" if vmax - vmin > 0.3 else "良好" if vmax - vmin > 0.15 else "一般"
        ax.text(0.98, 0.02, f'渐变质量: {gradient_quality}', transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none', format='png')
            plt.close()
            print(f"  植被渐变地图已保存: {output_path}")
        else:
            plt.show()
        
        return fig

    def test_gradient_visualization(self):
        """测试渐变可视化效果"""
        print("=== 测试NDVI渐变可视化效果 ===\n")
        
        # 加载数据
        ndvi_data = self.load_multiple_years_ndvi()
        
        if not ndvi_data:
            print("没有找到有效数据")
            return
        
        print(f"\n成功加载 {len(ndvi_data)} 个月的数据")
        
        # 分析所有数据以了解整体范围
        all_valid_data = []
        for month, data in ndvi_data.items():
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                all_valid_data.extend(valid.flatten())
        
        if all_valid_data:
            global_min = np.min(all_valid_data)
            global_max = np.max(all_valid_data)
            global_mean = np.mean(all_valid_data)
            global_std = np.std(all_valid_data)
            
            print(f"\n全年数据总体统计:")
            print(f"  全局范围: {global_min:.4f} 到 {global_max:.4f}")
            print(f"  全局均值: {global_mean:.4f}, 标准差: {global_std:.4f}")
            
            # 检查数据变化是否足够
            if global_max - global_min < 0.05:
                print(f"    警告: 数据变化范围很小 ({global_max - global_min:.4f})")
                print("  这可能解释了为什么渐变效果不明显")
            else:
                print(f"   数据有足够的变化范围用于渐变可视化")
        
        # 测试几个月份的可视化，选择变化最大的月份
        monthly_ranges = {}
        for month, data in ndvi_data.items():
            valid = data[~np.isnan(data)]
            if len(valid) > 10:  # 确保有足够的有效数据
                monthly_ranges[month] = np.max(valid) - np.min(valid)
        
        # 选择变化最大的3个月进行测试
        if monthly_ranges:
            top_months = sorted(monthly_ranges.items(), key=lambda x: x[1], reverse=True)[:3]
            test_months = [month for month, _ in top_months]
            
            print(f"\n选择变化最大的月份进行测试:")
            for month, range_val in top_months:
                print(f"  {self.month_names[month]}: 变化范围 {range_val:.4f}")
        else:
            test_months = sorted(ndvi_data.keys())[:3]
        
        for month in test_months:
            print(f"\n--- 处理 {self.month_names[month]} ({month}月) ---")
            output_file = f'gradient_test_{month}_{self.month_names[month].lower()}.png'
            self.create_enhanced_gradient_map(ndvi_data[month], month, output_file)
        
        print(f"\n 渐变测试完成！生成了 {len(test_months)} 个测试图像")
        
        # 提供数据改进建议
        if all_valid_data and global_max - global_min < 0.1:
            print(f"\n 改进建议:")
            print(f"  1. 数据范围较小，考虑使用更敏感的颜色映射")
            print(f"  2. 可以尝试局部区域分析而不是全球视图")
            print(f"  3. 使用对比度增强或数据标准化技术")

    def create_full_animation_with_gradient(self, output_path='gradient_fixed_ndvi_2024.gif'):
        """创建具有良好渐变效果的完整动画"""
        frames = []
        months = sorted(self.ndvi_data.keys())
        
        print(f"\n创建渐变优化的动画，包含 {len(months)} 个月...")
        
        for i, month in enumerate(months):
            print(f"处理 {self.month_names[month]} ({i+1}/{len(months)})")
            
            temp_path = f'temp_gradient_{month}.png'
            self.create_enhanced_gradient_map(self.ndvi_data[month], month, temp_path)
            
            if os.path.exists(temp_path):
                frame = imageio.imread(temp_path)
                frames.append(frame)
                os.remove(temp_path)
        
        if frames:
            print("生成GIF动画...")
            imageio.mimsave(output_path, frames, duration=2.0, loop=0)
            print(f" 渐变优化动画已保存: {output_path}")
        else:
            print(" 没有生成任何帧")
        
        return output_path

    def create_forced_gradient_map(self, ndvi_data, month, output_path=None):
        """强制创建渐变效果的地图，即使数据变化很小"""
        
        print(f"\n=== 强制渐变模式 - {self.month_names[month]} ===")
        
        # 分析数据
        valid_data = ndvi_data[~np.isnan(ndvi_data)]
        if len(valid_data) == 0:
            print(f"警告: {self.month_names[month]}没有有效数据")
            return None
        
        data_min = np.nanmin(valid_data)
        data_max = np.nanmax(valid_data)
        data_mean = np.nanmean(valid_data)
        
        # 强制扩展数据范围以产生渐变效果
        # 方法1：数据拉伸
        stretched_data = ndvi_data.copy()
        
        # 对有效数据进行线性拉伸到更大范围
        valid_mask = ~np.isnan(stretched_data)
        if np.any(valid_mask):
            valid_vals = stretched_data[valid_mask]
            
            # 拉伸到 -0.2 到 0.8 的范围
            if data_max > data_min:
                stretched_vals = (valid_vals - data_min) / (data_max - data_min)
                stretched_vals = stretched_vals * 1.0 - 0.2  # 映射到 -0.2 到 0.8
                stretched_data[valid_mask] = stretched_vals
            
            print(f"数据拉伸: 原始范围 {data_min:.4f}-{data_max:.4f} → 拉伸范围 {np.nanmin(stretched_data):.4f}-{np.nanmax(stretched_data):.4f}")
        
        # 设置图形
        fig = plt.figure(figsize=(18, 10), dpi=150)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        
        # 地图特征
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='black', alpha=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='gray', alpha=0.6)
        ax.add_feature(cfeature.LAND, facecolor='#f8f8f8', alpha=0.1)
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff', alpha=0.2)
        
        # 坐标网格
        lon = np.linspace(-179.5, 179.5, stretched_data.shape[1])
        lat = np.linspace(89.5, -89.5, stretched_data.shape[0])
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # 使用拉伸后的数据范围
        vmin = np.nanmin(stretched_data)
        vmax = np.nanmax(stretched_data)
        
        # 创建高对比度颜色映射
        colors = [
            '#4A0E0E',  # 深暗红
            '#8B0000',  # 深红
            '#CD5C5C',  # 印度红
            '#F4A460',  # 沙棕
            '#DEB887',  # 浅黄褐
            '#F0E68C',  # 卡其
            '#FFFF00',  # 亮黄
            '#ADFF2F',  # 绿黄
            '#7FFF00',  # 草绿
            '#32CD32',  # 酸橙绿
            '#228B22',  # 森林绿
            '#006400',  # 深绿
            '#004000',  # 极深绿
        ]
        
        forced_cmap = LinearSegmentedColormap.from_list('forced_gradient', colors, N=512)
        
        # 使用高密度levels确保平滑渐变
        levels = np.linspace(vmin, vmax, 100)
        
        masked_data = np.ma.masked_invalid(stretched_data)
        
        im = ax.contourf(lon_grid, lat_grid, masked_data,
                        levels=levels,
                        cmap=forced_cmap,
                        extend='both',
                        transform=ccrs.PlateCarree(),
                        alpha=0.95,
                        antialiased=True)
        
        # 增强的颜色条
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal',
                           pad=0.06, shrink=0.8, aspect=50)
        cbar.set_label('Enhanced NDVI Gradient (Stretched Values)', 
                      fontsize=14, fontweight='bold', labelpad=12)
        
        # 显示原始数据范围和拉伸范围
        tick_values = np.linspace(vmin, vmax, 9)
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f'{val:.2f}' for val in tick_values])
        cbar.ax.tick_params(labelsize=11)
        
        # 标题
        title = f'Enhanced Gradient NDVI - {self.month_names[month]} 2024'
        plt.title(title, fontsize=20, fontweight='bold', pad=25,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
        
        # 数据信息
        valid_count = len(valid_data)
        total_count = ndvi_data.size
        valid_percentage = (valid_count / total_count) * 100
        
        info_text = (f'Data Coverage: {valid_percentage:.1f}%\n'
                    f'Original Range: {data_min:.4f} - {data_max:.4f}\n'
                    f'Stretched Range: {vmin:.2f} - {vmax:.2f}\n'
                    f'Enhancement: Forced Gradient Mode')
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none', format='png')
            plt.close()
            print(f"  强制渐变地图已保存到: {output_path}")
        else:
            plt.show()
        
        return fig

    def create_successful_gradient_animation(self, output_path='successful_gradient_ndvi_2024.gif'):
        """基于成功的强制渐变方法创建完整动画"""
        print(f"\n=== 创建成功渐变效果动画 ===")
        
        # 确保数据已加载
        if not self.ndvi_data:
            print("正在加载数据...")
            self.load_multiple_years_ndvi()
            
        if not self.ndvi_data:
            print(" 没有找到有效数据")
            return None
            
        frames = []
        months = sorted(self.ndvi_data.keys())
        
        print(f"使用成功的渐变方法创建动画，包含 {len(months)} 个月...")
        
        for i, month in enumerate(months):
            print(f"处理 {self.month_names[month]} ({i+1}/{len(months)})")
            
            temp_path = f'temp_successful_gradient_{month}.png'
            
            # 使用成功的强制渐变方法
            self.create_forced_gradient_map(self.ndvi_data[month], month, temp_path)
            
            if os.path.exists(temp_path):
                frame = imageio.imread(temp_path)
                frames.append(frame)
                os.remove(temp_path)
                print(f"   已处理 {self.month_names[month]}")
            else:
                print(f"   处理失败 {self.month_names[month]}")
        
        if frames:
            print(f"\n生成GIF动画（3秒每帧）...")
            imageio.mimsave(output_path, frames, duration=3.0, loop=0, quality=10)
            
            # 检查文件大小
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024*1024)  # MB
                print(f" 成功渐变动画已保存!")
                print(f"   文件: {output_path}")
                print(f"   大小: {file_size:.1f} MB")
                print(f"   帧数: {len(frames)}")
                print(f"   月份: {[self.month_names[m] for m in months]}")
                return output_path
            else:
                print(" 动画文件保存失败")
                return None
        else:
            print(" 没有生成任何帧")
            return None

def main():
    """主函数"""
    print("=== 渐变修复版NDVI可视化系统 ===")
    
    data_dir = r'c:\Users\dell\Desktop\DM文件夹\6月6\小项目\高级可视化'
    visualizer = FixedGradientNDVIVisualizer(data_dir)
    
    print("\n选择操作模式:")
    print("1. 测试不同的99999填充值处理方法")
    print("2. 使用标准方法测试渐变效果")
    print("3. 强制渐变模式 (单张测试)")
    print("4. 创建成功渐变动画 (完整2024年)")
    
    choice = input("请选择 (1-4): ").strip()
    
    if choice == '1':
        # 测试不同填充方法
        visualizer.test_different_fill_methods()
        
    elif choice == '2':
        # 标准测试
        visualizer.test_gradient_visualization()
        
    elif choice == '3':
        # 强制渐变模式
        ndvi_data = visualizer.load_multiple_years_ndvi()
        if ndvi_data:
            test_month = sorted(ndvi_data.keys())[0]
            output_file = f'forced_gradient_{test_month}.png'
            visualizer.create_forced_gradient_map(ndvi_data[test_month], test_month, output_file)
            
    elif choice == '4':
        # 创建成功渐变动画
        print("基于成功的forced_gradient方法创建完整动画...")
        animation_path = visualizer.create_successful_gradient_animation()
        if animation_path:
            print(f"\n 动画创建成功！文件路径: {animation_path}")
        else:
            print("\n 动画创建失败")

if __name__ == "__main__":
    main()
