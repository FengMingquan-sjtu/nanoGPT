# fuser -v /dev/nvidia*
# pkill -f python3.13
# pkill -f train.py


CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 2 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --compile=True --wandb_run_name='cont-qwen2-1.5B-finewebedu-0.8rho' --out_dir='out-prodcpfs/cont-qwen2-1.5B-finewebedu-0.8rho' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-1.5B' --ref_model_ckpt='/prodcpfs/user/fengmingquan/model/Qwen2-1.5B-Instruct'   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --batch_size=1 --gradient_accumulation_steps=480 --block_size=4096 --token_keep_ratio=0.8 --max_iters=20000 --lr_decay_iters=20000 --learning_rate=7e-6 --min_lr=7e-7  > log/gpt-owm-rho-0.log 2>&1 &



CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 2 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=False --compile=True --wandb_run_name='cont-qwen2.5-3B-finewebedu' --out_dir='out-prodcpfs/cont-qwen2.5-3B-finewebedu' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2.5-3B' --ref_model_ckpt=""  --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --batch_size=1 --gradient_accumulation_steps=480 --block_size=4096 --token_keep_ratio=1.0 --max_iters=20000 --lr_decay_iters=20000 --learning_rate=7e-6 --min_lr=7e-7  > log/gpt-owm-rho-0.log 2>&1 &



CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 2 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=False --compile=True --wandb_run_name='tmp' --out_dir='out-prodcpfs/tmp' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt=''   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --batch_size=2 --gradient_accumulation_steps=240 --block_size=4096 --token_keep_ratio=1.0 --max_iters=20000 --lr_decay_iters=20000 --learning_rate=7e-6 --min_lr=7e-7  > log/gpt-owm-rho-0.log 2>&1 &


nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/deepspeed --num_gpus 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='cont-qwen2-0.5B-finewebedu-0.8rho-1.5Bref' --out_dir='out-prodcpfs/cont-qwen2-0.5B-finewebedu-0.8rho-1.5Bref' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt='/prodcpfs/user/fengmingquan/model/Qwen2-1.5B'   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --batch_size=4 --gradient_accumulation_steps=120 --block_size=4096 --token_keep_ratio=0.8 --max_iters=20000 --lr_decay_iters=20000 --learning_rate=7e-6 --min_lr=7e-7  > log/gpt-owm-rho-1.log 2>&1 &

OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='cont-qwen2.5-3B-finewebedu-0.8rho' --out_dir='out-prodcpfs/cont-qwen2.5-3B-finewebedu-0.8rho' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2.5-3B' --ref_model_ckpt='/prodcpfs/user/fengmingquan/model/Qwen2.5-3B-Instruct'   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --batch_size=1 --gradient_accumulation_steps=480 --block_size=4096 --token_keep_ratio=0.8 --max_iters=20000 --lr_decay_iters=20000 --learning_rate=7e-6 --min_lr=7e-7  > log/gpt-owm-rho-2.log 2>&1 &

OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=False --use_deepspeed=True --zero_stage=2 --wandb_run_name='cont-qwen2.5-3B-finewebedu-0.8rho' --out_dir='out-prodcpfs/tmp' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2.5-3B' --ref_model_ckpt='/prodcpfs/user/fengmingquan/model/Qwen2.5-3B-Instruct'   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --batch_size=1 --gradient_accumulation_steps=480 --block_size=40 --token_keep_ratio=0.8 --max_iters=20000 --lr_decay_iters=20000 --learning_rate=7e-6 --min_lr=7e-7  > log/gpt-owm-rho-3.log 2>&1 &


cd /cpfs/user/fengmingquan/nanoGPT

ali-sg-acr-registry-vpc.ap-southeast-1.cr.aliyuncs.com/xhs-llm/xhs-llm:ngc-2403-taishan2

export PATH=$PATH:/cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin
torchrun --standalone --nproc_per_node=8 train_nanogpt.py config/train_arithmetic_char.py  --gradient_accumulation_steps=8 --batch_size=8


juicefs sync -u -p 100 --exclude "00*.parquet" oss://LTAI5tDuCoTTh6gu5PK8gFfN:6eZGIEeLo81eRSqYRMHUEG2FcVTkQV@lsg-oss-chatgpt-agi-hcfs.oss-ap-southeast-1-internal.aliyuncs.com/crawl/multimodal/HuggingFaceFW/fineweb-edu/sample/100BT/ /prodcpfs/user/fengmingquan/dataset/raw/fineweb-edu/sample/100BT-25BT/