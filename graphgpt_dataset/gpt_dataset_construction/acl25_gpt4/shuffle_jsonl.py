import pandas as pd

# 读取 JSONL 文件
df = pd.read_json('acl_conversations.jsonl', lines=True)

# 打乱数据
shuffled_df = df.sample(frac=1).reset_index(drop=True)

# 将打乱后的数据保存为新的 JSONL 文件
shuffled_df.to_json('shuffled_acl_conversations.jsonl', orient='records', lines=True)