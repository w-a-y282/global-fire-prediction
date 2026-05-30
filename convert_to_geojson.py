import json
import os
import shutil
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm

# ===================== 核心配置 =====================
MAX_PROCESSES = 3
WRITE_BATCH = 5_000_000
COORD_PREC = 2
# 🔥 预测值过滤阈值（预测值 < 此值 且 真实标签=0 的数据将被剔除）
PRED_FILTER_THRESHOLD = 0.1
# 🔥 关闭采样，展示全部有效数据
SAMPLE_RATIO = 1.0

OUT_DIR = "fire_data_by_date"
BATCH_TMP_DIR = "batch_tmp"

ALL_REGIONS = [
    (1, "fb_1111_2021"),
    (2, "fb_2222_2021"),
    (3, "fb_3333_2021"),
    (4, "fb_4444_2021"),
    (5, "fb_5555_2021"),
]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(BATCH_TMP_DIR, exist_ok=True)


def process_single_region(args):
    rid, folder = args
    csv_path = os.path.join(folder, "draw_filtered.csv")

    if not os.path.exists(csv_path):
        print(f"⚠️  区域{rid} 文件不存在，跳过")
        return

    print(f"\n🚀 进程启动：开始处理区域 {rid}")
    print(f"🔍 过滤规则：预测值 < {PRED_FILTER_THRESHOLD} 且 真实标签=0 的数据将被剔除")

    region_tmp_dir = os.path.join(BATCH_TMP_DIR, f"region_{rid}")
    os.makedirs(region_tmp_dir, exist_ok=True)

    date_cache = defaultdict(dict)
    processed = 0
    filtered = 0
    batch_num = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        header = [h.strip() for h in header]

        try:
            di = header.index("date")
        except:
            di = header.index("dt")
        li = header.index("lon")
        lati = header.index("lat")
        labi = header.index("label")

        predi = None
        for col in ["pred", "pred_fb", "pred_xgb", "pred_final", "pre"]:
            if col in header:
                predi = header.index(col)
                break
        if predi is None:
            print(f"❌ 区域{rid} 找不到预测列")
            return

        pbar = tqdm(desc=f"区域{rid}", unit="行", leave=True)

        for line in f:
            row = line.split(",")
            try:
                d = row[di][:10]
                lon = round(float(row[li]), COORD_PREC)
                lat = round(float(row[lati]), COORD_PREC)
                lab = float(row[labi])
                pre = float(row[predi])

                # 🔥 核心过滤逻辑
                if pre < PRED_FILTER_THRESHOLD and lab == 0.0:
                    filtered += 1
                    processed += 1
                    pbar.update(1)
                    continue

                key = f"{lon},{lat}"
                if key not in date_cache[d]:
                    date_cache[d][key] = {
                        "label_sum": lab,
                        "pred_sum": pre,
                        "count": 1,
                        "rid": rid
                    }
                else:
                    date_cache[d][key]["label_sum"] += lab
                    date_cache[d][key]["pred_sum"] += pre
                    date_cache[d][key]["count"] += 1

            except:
                processed += 1
                pbar.update(1)
                continue

            processed += 1
            pbar.update(1)

            if processed % WRITE_BATCH == 0:
                batch_num += 1
                batch_file = os.path.join(region_tmp_dir, f"batch_{batch_num}.json")
                with open(batch_file, "w", encoding="utf-8") as bf:
                    json.dump(date_cache, bf, separators=(",", ":"))
                date_cache.clear()
                filter_rate = filtered / processed * 100
                print(f"\n✅ 区域{rid} 批次{batch_num} 写入完成，过滤率{filter_rate:.2f}%")

        pbar.close()

    if date_cache:
        batch_num += 1
        batch_file = os.path.join(region_tmp_dir, f"batch_{batch_num}.json")
        with open(batch_file, "w", encoding="utf-8") as bf:
            json.dump(date_cache, bf, separators=(",", ":"))
        del date_cache

    filter_rate = filtered / processed * 100 if processed > 0 else 0
    print(f"\n🎉 区域{rid} 处理完成！")
    print(f"📊 总计处理{processed:,}行，过滤{filtered:,}行，过滤率{filter_rate:.2f}%")


def merge_all_batches():
    print("\n" + "=" * 60)
    print("🚀 开始合并所有区域数据")
    print("=" * 60)

    final_data = defaultdict(dict)

    for region_dir in os.listdir(BATCH_TMP_DIR):
        region_path = os.path.join(BATCH_TMP_DIR, region_dir)
        if not os.path.isdir(region_path):
            continue

        rid = int(region_dir.replace("region_", ""))
        print(f"\n📥 正在合并区域 {rid} 的数据...")

        for batch_file in tqdm(os.listdir(region_path), desc=f"区域{rid}合并"):
            if not batch_file.endswith(".json"):
                continue
            bpath = os.path.join(region_path, batch_file)

            with open(bpath, "r", encoding="utf-8") as f:
                batch_data = json.load(f)

            for dt, points in batch_data.items():
                for coord, val in points.items():
                    if coord not in final_data[dt]:
                        final_data[dt][coord] = val
                    else:
                        final_data[dt][coord]["label_sum"] += val["label_sum"]
                        final_data[dt][coord]["pred_sum"] += val["pred_sum"]
                        final_data[dt][coord]["count"] += val["count"]

    print("\n✅ 所有数据合并完成！")
    return final_data


def generate_geojson_files(final_data):
    print("\n" + "=" * 60)
    print("🗺️  开始生成GeoJSON文件")
    print("=" * 60)

    date_list = sorted(final_data.keys())

    for dt in tqdm(date_list, desc="生成GeoJSON"):
        points = final_data[dt]
        features = []

        for pos, val in points.items():
            try:
                lon, lat = map(float, pos.split(","))
                avg_pred = val["pred_sum"] / val["count"]
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "region": val["rid"],
                        "fire_occurred": 1 if val["label_sum"] > 0 else 0,
                        "prediction": round(avg_pred, 4)  # 🔥 只保留真实的当天预测值
                    }
                })
            except:
                continue

        out_path = os.path.join(OUT_DIR, f"fire_{dt}.geojson")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f, separators=(",", ":"))

    # 保存日期列表
    with open(os.path.join(OUT_DIR, "date_list.json"), "w", encoding="utf-8") as f:
        json.dump(date_list, f)

    # 清理临时文件
    shutil.rmtree(BATCH_TMP_DIR)
    print(f"\n🎉 全部完成！")
    print(f"📊 共生成 {len(date_list)} 个日期的GeoJSON文件")
    print(f"📁 所有文件保存在: {OUT_DIR}/")
    print(f"📅 日期范围: {date_list[0]} 至 {date_list[-1]}")


if __name__ == "__main__":
    print("=" * 60)
    print("🔥 火灾数据GeoJSON生成工具（真实预测版）")
    print("=" * 60)
    print(f"⚙️  配置：同时处理{MAX_PROCESSES}个区域")
    print(f"🔍 过滤阈值：预测值 < {PRED_FILTER_THRESHOLD} 且 真实标签=0")
    print(f"📊 采样比例：{SAMPLE_RATIO*100}%（全部保留）")
    print("=" * 60)

    with Pool(processes=MAX_PROCESSES) as pool:
        pool.map(process_single_region, ALL_REGIONS)

    final_data = merge_all_batches()
    generate_geojson_files(final_data)
