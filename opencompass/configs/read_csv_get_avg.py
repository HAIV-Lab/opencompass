
import pandas as pd

# 1. 读取 CSV 文件
df = pd.read_csv('/kpfs/user/wtt/opencompass/results/ceval/qwen3/4B/thinking/20260615_144725/summary/summary_20260615_144725.csv')

# 2. 获取最后一列的列名（这里是 'vllm_qwen3_8b_no_thinking'）
last_column_name = df.columns[-1]
valid_data = df[last_column_name].dropna()  # 去除NaN值
count = len(valid_data) 
# 3. 计算该列的平均值
average_value = valid_data.mean()

print(f"最后一列 ({last_column_name}) 的平均值是: {average_value:.2f}")
print(f"共计算了 {count} 个值")