import pandas as pd
import json
import os
import warnings

warnings.filterwarnings('ignore')

# 区域配置
regions = [
    {"id": 1, "fb_folder": "fb_1111_2021", "xgb_folder": "xg_1111_2021"},
    {"id": 2, "fb_folder": "fb_2222_2021", "xgb_folder": "xg_2222_2021"},
    {"id": 3, "fb_folder": "fb_3333_2021", "xgb_folder": "xg_3333_2021"},
    {"id": 4, "fb_folder": "fb_4444_2021", "xgb_folder": "xg_4444_2021"},
    {"id": 5, "fb_folder": "fb_5555_2021", "xgb_folder": "xg_5555_2021"}
]

all_features = []

for region in regions:
    region_id = region["id"]
    print(f"\n处理区域 {region_id}...")

    # 读取FB模型数据
    fb_file = os.path.join(region["fb_folder"], "draw.csv")
    if os.path.exists(fb_file):
        print(f"  读取FB模型: {fb_file}")
        # 使用pandas快速读取CSV
        df_fb = pd.read_csv(
            fb_file,
            usecols=['lon', 'lat', 'label', 'pred'],  # 只读取需要的列
            dtype={
                'lon': 'float32',
                'lat': 'float32',
                'label': 'float32',
                'pred': 'float32'
            }
        )
        print(f"  共 {len(df_fb)} 行数据")

        # 降低坐标精度到3位小数(约100米精度，完全足够)
        df_fb['lon'] = df_fb['lon'].round(3)
        df_fb['lat'] = df_fb['lat'].round(3)

        # 按经纬度聚合(C语言实现，速度极快)
        agg_fb = df_fb.groupby(['lon', 'lat']).agg({
            'label': ['sum', 'count'],
            'pred': 'sum'
        }).reset_index()

        # 重命名列
        agg_fb.columns = ['lon', 'lat', 'fire_days', 'count', 'fb_pre_sum']

        # 释放内存
        del df_fb

    # 读取XGB模型数据
    xgb_file = os.path.join(region["xgb_folder"], "draw.csv")
    if os.path.exists(xgb_file):
        print(f"  读取XGB模型: {xgb_file}")
        df_xgb = pd.read_csv(
            xgb_file,
            usecols=['lon', 'lat', 'pred'],
            dtype={
                'lon': 'float32',
                'lat': 'float32',
                'pred': 'float32'
            }
        )
        print(f"  共 {len(df_xgb)} 行数据")

        # 同样降低精度
        df_xgb['lon'] = df_xgb['lon'].round(3)
        df_xgb['lat'] = df_xgb['lat'].round(3)

        # 聚合
        agg_xgb = df_xgb.groupby(['lon', 'lat']).agg({
            'pred': 'sum'
        }).reset_index()

        agg_xgb.columns = ['lon', 'lat', 'xgb_pre_sum']

        # 释放内存
        del df_xgb

    # 合并FB和XGB数据
    print(f"  合并数据...")
    merged = pd.merge(agg_fb, agg_xgb, on=['lon', 'lat'], how='outer')

    # 填充缺失值
    merged['fire_days'] = merged['fire_days'].fillna(0).astype(int)
    merged['count'] = merged['count'].fillna(0).astype(int)
    merged['fb_pre_sum'] = merged['fb_pre_sum'].fillna(0)
    merged['xgb_pre_sum'] = merged['xgb_pre_sum'].fillna(0)

    # 计算平均值
    merged['fb_pre_avg'] = merged['fb_pre_sum'] / merged['count']
    merged['xgb_pre_avg'] = merged['xgb_pre_sum'] / merged['count']

    # 添加区域信息
    merged['region'] = region_id

    # 转换为GeoJSON格式
    print(f"  生成GeoJSON要素...")
    features = []
    for _, row in merged.iterrows():
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [row['lon'], row['lat']]
            },
            "properties": {
                "region": region_id,
                "fire_days": int(row['fire_days']),
                "fb_pre_avg": float(row['fb_pre_avg']),
                "xgb_pre_avg": float(row['xgb_pre_avg'])
            }
        }
        features.append(feature)

    all_features.extend(features)
    print(f"  区域 {region_id} 完成，生成 {len(features)} 个点")

# 创建最终GeoJSON
geojson = {
    "type": "FeatureCollection",
    "features": all_features
}

# 保存文件
output_file = "fire_data.geojson"
print(f"\n保存文件到 {output_file}...")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False)

file_size = os.path.getsize(output_file) / 1024 / 1024
print(f"\n转换完成！")
print(f"共生成 {len(all_features)} 个唯一地理点")
print(f"文件大小: {file_size:.2f} MB")

# 统计信息
if all_features:
    max_fire = max(f["properties"]["fire_days"] for f in all_features)
    avg_fire = sum(f["properties"]["fire_days"] for f in all_features) / len(all_features)
    print(f"\n数据统计:")
    print(f"  最大火灾天数: {max_fire} 天")
    print(f"  平均火灾天数: {avg_fire:.2f} 天")