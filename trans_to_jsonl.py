import pandas as pd
import json
import os
from sklearn.preprocessing import StandardScaler

def convert_csv_to_jsonl_with_scaling(n_points,
                                      csv_path,
                                       output_jsonl_path=None, 
                                       exclude_columns=None):
    """
    将 CSV 中每个数值列做 StandardScaler 标准化后，转为单变量 JSONL 格式。

    参数：
        csv_path (str): 输入 CSV 文件路径
        output_jsonl_path (str): 输出 JSONL 路径，默认与 CSV 同名
        exclude_columns (list[str]): 要排除的列名（如 ['date', 'timestamp']）
    """
    # 1. 加载数据
    df = pd.read_csv(csv_path)
    print(f"✅ 加载 CSV：{csv_path}")
    print(f"📌 所有列：{df.columns.tolist()}")

    #提取前n行数据
    n = n_points
    df = df.head(n)
    print(f"已保留前 {n} 行数据")

    # 2. 设置排除列
    if exclude_columns is None:
        exclude_columns = ['date', 'datetime', 'timestamp', 'time']

    # 3. 获取需要标准化的数值列
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    variable_columns = [col for col in numeric_columns if col.lower() not in exclude_columns]
    print(f"🔢 标准化列：{variable_columns}")

    if not variable_columns:
        raise ValueError("❌ 未找到可用的数值列。")

    # 4. 标准化
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[variable_columns] = scaler.fit_transform(df_scaled[variable_columns])

    # 5. 设置 JSONL 输出路径
    if output_jsonl_path is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_jsonl_path = os.path.join(os.path.dirname(csv_path), base_name + '_scaled.jsonl')

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    # 6. 写入为 JSONL，每个变量一行
    with open(output_jsonl_path, 'w') as f:
        for col in variable_columns:
            sequence = df_scaled[col].dropna().tolist()
            json_line = json.dumps({'sequence': sequence})
            f.write(json_line + '\n')

    print(f"✅ 已生成标准化 JSONL 文件：{output_jsonl_path}")

# ✅ 示例用法
convert_csv_to_jsonl_with_scaling(
    n_points=8640,
    csv_path='eval_data/ETTh1.csv',
    output_jsonl_path='train_data/ETTh1_train.jsonl'
)
