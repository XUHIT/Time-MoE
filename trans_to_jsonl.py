import pandas as pd
import json
import os

# 1. 读取上传的CSV文件
file_path = r'/home/super/git_projects/Time-MoE-main/eval_data/ETTh1.csv'  # 你的文件路径
df = pd.read_csv(file_path)

# 2. 查看数据结构，确认数据的列名
print(df.head())  # 显示前几行数据，检查数据结构

# 3. 定义需要处理的变量列（排除时间戳列等非数值列）
variable_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']  # ETTh1 的变量列

# 4. 创建输出目录和 JSONL 文件路径
output_dir = r'/home/super/git_projects/Time-MoE-main/data'
os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
jsonl_file_path = os.path.join(output_dir, 'ETTh1.jsonl')

# 5. 打开 JSONL 文件进行写入
with open(jsonl_file_path, 'w') as jsonl_file:
    # 6. 为每个变量生成单变量时间序列
    for column in variable_columns:
        sequence = df[column].tolist()  # 获取该列的所有值作为时间序列
        data = {'sequence': sequence}  # 创建字典对象，包含单变量序列
        # 7. 将字典对象转换为 JSON 格式并写入 JSONL 文件
        jsonl_file.write(json.dumps(data) + '\n')

# 8. 打印生成的 JSONL 文件路径
print(f'Generated JSONL file: {jsonl_file_path}')