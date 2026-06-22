export OPENCOMPASS_DATA_ROOT=/kpfs/user/wtt/opencompass/data
python run.py \
    --models hf_qwen2_5_7b_instruct \
    --datasets ceval_gen \
    --hf-type chat \
    --hf-path /kpfs/intern-legal/model/Qwen2.5-7B-Instruct \
    -w outputs/ceval/qwen2.5_7b_instruct \