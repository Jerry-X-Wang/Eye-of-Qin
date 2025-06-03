#summer is the -er form of sum, not the season summer
import json
import os
from collections import defaultdict
import pandas as pd

def aggregate_state_times(input_dir, output_file):
    # 初始化字典存储累计时间
    time_data = defaultdict(lambda: {'awake': 0.0, 'sleeping': 0.0})
    
    # 遍历输入目录下的所有JSON文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    user_id = entry['id']
                    time_data[user_id]['awake'] += entry['state_time']['awake']
                    time_data[user_id]['sleeping'] += entry['state_time']['sleeping']
    
    # 转换为DataFrame并排序
    df = pd.DataFrame.from_dict(time_data, orient='index').reset_index()
    df.columns = ['ID', 'Awake Time', 'Sleeping Time']
    df = df.sort_values('ID')
    
    # 保存为Excel文件
    df.to_excel(output_file, index=False)
    print(f"文件已生成：{output_file}")

# 使用示例
if __name__ == "__main__":
    aggregate_state_times(
        input_dir='data/time',
        output_file='time_sum.xlsx'
    )
