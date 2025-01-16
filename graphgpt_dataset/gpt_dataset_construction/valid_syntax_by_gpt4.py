import pandas as pd
from pathlib import Path
import subprocess
import json

file_path = "./"
file_name = "gpt4_graphs2.jsonl"
file = Path(file_path).joinpath(file_name)
df = pd.read_json(file, lines=True)
command = ["slang-driver", "code.v", "--ast-json", "code.json"]
for index, row in df.iterrows():
    print(index)
    content = row['Response']
    with open('./code.v', 'w') as f:
        f.write(content)
    try:
    # 调用外部命令
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        
        # 如果成功，输出结果
        print("命令执行成功:")
        print(result.stdout)
        ##write row to jsonl
        row_dict = row.to_dict()
        with open('valid_multi_module.jsonl', 'a') as f:
            f.write(json.dumps(row_dict) + '\n')
        print(index)
        
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，输出错误信息
        print("命令执行失败:")
        print(e.stderr)

    # print(index)