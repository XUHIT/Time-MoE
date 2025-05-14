import json

file_path = 'train_data/ETTh1_train.jsonl'  # 修改为你的 jsonl 路径

lengths = []

with open(file_path, 'r') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            seq = data['sequence']
            lengths.append(len(seq))
        except Exception as e:
            print(f"[ERROR] Line {i} 格式错误：{e}")

# 打印统计信息
print(f"总样本数：{len(lengths)}")
print(f"最大长度：{max(lengths)}")
print(f"最小长度：{min(lengths)}")
print(f"平均长度：{sum(lengths)/len(lengths):.2f}")

