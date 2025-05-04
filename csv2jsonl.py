import pandas as pd
import json
import argparse

def csv_to_jsonl(input_csv_path, output_jsonl_path, num_rows=None):
    """
    将 CSV 文件中的每一列（除 date）作为一个时间序列，写入 JSONL 文件。
    每列进行 Z-score 标准化，输出格式为：
    {"sequence": [标准化后的值1, 值2, ..., 值8640]}
    """

    print(f"📥 正在读取 CSV 文件：{input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"✅ 原始数据读取完成，共 {df.shape[0]} 行，{df.shape[1]} 列")

    # 去除 'date' 列（不区分大小写）
    original_columns = df.columns.tolist()
    df = df[[col for col in df.columns if col.lower() != 'date']]
    removed = set(original_columns) - set(df.columns.tolist())
    if removed:
        print(f"🧹 已移除列：{removed}")

    # 截取前 num_rows 行（如设置）
    if num_rows is not None:
        df = df.head(num_rows)
        print(f"✂️ 已截取前 {num_rows} 行数据")

    # 打印每列非空值数量
    print("📈 每列非空值数量如下：")
    for col in df.columns:
        print(f"{col}: {df[col].count()}")

    # 执行 Z-score 标准化
    print("📊 正在执行 Z-score 标准化...")
    df_standardized = (df - df.mean()) / df.std()

    # 检查 NaN（如某列 std 为 0）
    if df_standardized.isnull().values.any():
        print("❌ 错误：标准化结果中存在 NaN，以下是每列是否存在 NaN：")
        print(df_standardized.isnull().any())
        raise ValueError("标准化过程中出现 NaN，可能是某列标准差为 0。请检查数据。")

    # 写入 JSONL 文件（按列写，每列为一个序列）
    print(f"📝 正在写入 JSONL 文件到：{output_jsonl_path}")
    with open(output_jsonl_path, 'w') as f_out:
        for col in df_standardized.columns:
            sequence = df_standardized[col].tolist()
            json_line = json.dumps({"sequence": sequence})
            f_out.write(json_line + '\n')
        print(f"✅ 写入完成，共写入 {len(df_standardized.columns)} 条序列")

    print("🎉 全部处理完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 CSV 每列转换为 JSONL 序列（Z-score 标准化）")
    parser.add_argument('--input', type=str, default='eval_data/ETTh1.csv', help='输入的 CSV 文件路径')
    parser.add_argument('--output', type=str, default='train_data/ETTh1.jsonl', help='输出的 JSONL 文件路径')
    parser.add_argument('--num_rows', type=int, default=8640, help='只保留前 num_rows 行')

    args = parser.parse_args()
    csv_to_jsonl(args.input, args.output, args.num_rows)
