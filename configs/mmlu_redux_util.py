
from datasets import load_from_disk

path="/home/jovyan/.cache/opencompass/data/mmlu_redux/world_religions/data-00000-of-00001.arrow"
dataset = load_from_disk(path)
print(dataset)

