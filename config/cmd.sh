# fuser -v /dev/nvidia*
# pkill -f python3.13
# pkill -f train.py


CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 2 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --compile=True --wandb_run_name='cont-qwen2-1.5B-finewebedu-0.8rho' --out_dir='out-prodcpfs/cont-qwen2-1.5B-finewebedu-0.8rho' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-1.5B' --ref_model_ckpt='/prodcpfs/user/fengmingquan/model/Qwen2-1.5B-Instruct'   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --batch_size=1 --gradient_accumulation_steps=480 --block_size=4096 --token_keep_ratio=0.8 --max_iters=20000 --lr_decay_iters=20000 --learning_rate=7e-6 --min_lr=7e-7  > log/gpt-owm-rho-0.log 2>&1 &



CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 2 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=False --compile=True --wandb_run_name='cont-qwen2.5-3B-finewebedu' --out_dir='out-prodcpfs/cont-qwen2.5-3B-finewebedu' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2.5-3B' --ref_model_ckpt=""  --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --batch_size=1 --gradient_accumulation_steps=480 --block_size=4096 --token_keep_ratio=1.0 --max_iters=20000 --lr_decay_iters=20000 --learning_rate=7e-6 --min_lr=7e-7  > log/gpt-owm-rho-0.log 2>&1 &



CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 2 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=False --compile=True --wandb_run_name='tmp' --out_dir='out-prodcpfs/tmp' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt=''   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --batch_size=2 --gradient_accumulation_steps=240 --block_size=4096 --token_keep_ratio=1.0 --max_iters=20000 --lr_decay_iters=20000 --learning_rate=7e-6 --min_lr=7e-7  > log/gpt-owm-rho-0.log 2>&1 &


nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/deepspeed --num_gpus 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='cont-qwen2-0.5B-finewebedu-0.8rho-1.5Bref' --out_dir='out-prodcpfs/cont-qwen2-0.5B-finewebedu-0.8rho-1.5Bref' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt='/prodcpfs/user/fengmingquan/model/Qwen2-1.5B'   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --batch_size=4 --gradient_accumulation_steps=120 --block_size=4096 --token_keep_ratio=0.8 --max_iters=20000 --lr_decay_iters=20000 --learning_rate=7e-6 --min_lr=7e-7  > log/gpt-owm-rho-1.log 2>&1 &

OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='cont-qwen2.5-3B-finewebedu-0.8rho' --out_dir='out-prodcpfs/cont-qwen2.5-3B-finewebedu-0.8rho' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2.5-3B' --ref_model_ckpt='/prodcpfs/user/fengmingquan/model/Qwen2.5-3B-Instruct'   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --batch_size=1 --gradient_accumulation_steps=480 --block_size=4096 --token_keep_ratio=0.8 --max_iters=20000 --lr_decay_iters=20000 --learning_rate=7e-6 --min_lr=7e-7  > log/gpt-owm-rho-2.log 2>&1 &

OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=False --use_deepspeed=True --zero_stage=2 --wandb_run_name='' --out_dir='out-prodcpfs/tmp' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt=''   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --batch_size=2 --gradient_accumulation_steps=480 --block_size=40 --token_keep_ratio=0.8 --max_iters=20000 --lr_decay_iters=20000 --learning_rate=7e-6 --min_lr=7e-7  > log/gpt-owm-rho-3.log 2>&1 &

CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 4 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=0 --wandb_run_name='qwen2-0.5B-bio-raw' --out_dir='out-prodcpfs/qwen2-0.5B-bio-raw' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt=''   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --train_mode='scratch' --dataset_prefix="phys-llm-31-bio,phys-llm-31-qa" --dataset_ratio="50:0" --batch_size=48 --gradient_accumulation_steps=4 --block_size=512 --token_keep_ratio=1.0 --max_iters=40000 --lr_decay_iters=40000 --learning_rate=1e-3 --min_lr=1e-4 --eps=1e-6  > log/gpt-owm-rho-4.log 2>&1 &



OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='qwen2.5-3B-finewebedu' --out_dir='out-prodcpfs/qwen2.5-3B-finewebedu' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2.5-3B' --ref_model_ckpt=''   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --train_mode='scratch' --dataset_prefix="fineweb-edu-100bt,fineweb-edu-10bt" --dataset_ratio="100:0" --batch_size=2 --gradient_accumulation_steps=64 --block_size=4096 --token_keep_ratio=1.0 --max_iters=200000 --lr_decay_iters=200000 --learning_rate=6e-5 --min_lr=6e-6  > log/gpt-owm-rho-5.log 2>&1 &

OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='qwen2-0.5B-finewebedu-kinet' --out_dir='out-prodcpfs/qwen2-0.5B-finewebedu-kinet' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt=''   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --train_mode='scratch' --dataset_prefix="fineweb-edu-100bt,fineweb-edu-10bt" --dataset_ratio="100:0" --batch_size=6 --gradient_accumulation_steps=24 --block_size=4096 --token_keep_ratio=1.0 --max_iters=200000 --lr_decay_iters=200000 --learning_rate=1e-4 --min_lr=1e-5  > log/gpt-owm-rho-4.log 2>&1 &

OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='qwen2-1.5B-finewebedu-kinet' --out_dir='out-prodcpfs/qwen2-1.5B-finewebedu-kinet' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-1.5B' --ref_model_ckpt=''   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --train_mode='scratch' --dataset_prefix="fineweb-edu-100bt,fineweb-edu-10bt" --dataset_ratio="100:0" --batch_size=4 --gradient_accumulation_steps=32 --block_size=4096 --token_keep_ratio=1.0 --max_iters=200000 --lr_decay_iters=200000 --learning_rate=6e-5 --min_lr=6e-6  > log/gpt-owm-rho-6.log 2>&1 &

OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='qwen2-0.5B-finewebedu' --out_dir='out-prodcpfs/qwen2-0.5B-finewebedu' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt=''   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --train_mode='scratch' --dataset_prefix="fineweb-edu-100bt,fineweb-edu-10bt" --dataset_ratio="50:0" --batch_size=5 --gradient_accumulation_steps=24 --block_size=4096 --token_keep_ratio=1.0 --max_iters=200000 --lr_decay_iters=200000 --learning_rate=1e-4 --min_lr=1e-5 > log/gpt-owm-rho-5.log 2>&1 &

OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='qwen2-1.5B-finewebedu-distil-2.0-0.9-top50' --out_dir='out-prodcpfs/qwen2-1.5B-finewebedu-distil-2.0-0.9-top50' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-1.5B' --ref_model_ckpt='/prodcpfs/user/fengmingquan/model/Qwen2.5-3B'   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --train_mode='scratch' --loss_type='distill' --distill_ratio=0.9 --temperature=2.0 --distill_top_k=50 --dataset_prefix="fineweb-edu-100bt,fineweb-edu-10bt" --dataset_ratio="50:0" --batch_size=2 --gradient_accumulation_steps=64 --block_size=4096 --token_keep_ratio=1.0 --max_iters=200000 --lr_decay_iters=200000 --learning_rate=6e-5 --min_lr=6e-6 > log/gpt-owm-rho-8.log 2>&1 &

OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='qwen2-0.5B-finewebedu-distil-2.0-0.9-rkl' --out_dir='out-prodcpfs/qwen2-0.5B-finewebedu-distil-2.0-0.9-rkl' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt='/prodcpfs/user/fengmingquan/model/Qwen2-1.5B'   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --train_mode='scratch' --loss_type='distill' --div_mode="rkl" --distill_ratio=0.9 --temperature=2.0 --dataset_prefix="fineweb-edu-100bt,fineweb-edu-10bt" --dataset_ratio="50:0" --batch_size=3 --gradient_accumulation_steps=40 --block_size=4096 --token_keep_ratio=1.0 --max_iters=200000 --lr_decay_iters=200000 --learning_rate=6e-5 --min_lr=6e-6 > log/gpt-owm-rho-7.log 2>&1 &

OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='qwen2-0.5B-finewebedu-distil-2.0-0.9-top50' --out_dir='out-prodcpfs/qwen2-0.5B-finewebedu-distil-2.0-0.9-top50' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt=''   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --train_mode='resume' --loss_type='rho' --dataset_prefix="fineweb-edu-100bt,fineweb-edu-10bt" --dataset_ratio="50:0" --batch_size=6 --gradient_accumulation_steps=24 --block_size=4096 --token_keep_ratio=1.0 --max_iters=210000 --lr_decay_iters=210000 --learning_rate=6e-5 --min_lr=6e-6 > log/gpt-owm-rho-7.log 2>&1 &

OMP_NUM_THREADS=8 nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node 8 train.py config/cont_train_gpt2_owm_rho.py --wandb_log=True --use_deepspeed=True --zero_stage=2 --wandb_run_name='qwen2-0.5B-red' --out_dir='out-prodcpfs/qwen2-0.5B-red' --init_from='/prodcpfs/user/fengmingquan/model/Qwen2-0.5B' --ref_model_ckpt=''   --dataset='/prodcpfs/user/fengmingquan/dataset/processed-qwen2' --train_mode='scratch' --loss_type='distill' --dataset_prefix="red-1.0bt,fineweb-edu-10bt" --dataset_ratio="50:0" --batch_size=8 --gradient_accumulation_steps=16 --block_size=4096 --token_keep_ratio=1.0 --max_iters=12000 --lr_decay_iters=12000 --learning_rate=1e-4 --min_lr=1e-5 > log/gpt-owm-rho-4.log 2>&1 &

cd /cpfs/user/fengmingquan/nanoGPT

ali-sg-acr-registry-vpc.ap-southeast-1.cr.aliyuncs.com/xhs-llm/xhs-llm:ngc-2403-taishan2

export PATH=$PATH:/cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin
torchrun --standalone --nproc_per_node=8 train_nanogpt.py config/train_arithmetic_char.py  --gradient_accumulation_steps=8 --batch_size=8


juicefs sync -u -p 100 --exclude "00*.parquet" oss://LTAI5tDuCoTTh6gu5PK8gFfN:6eZGIEeLo81eRSqYRMHUEG2FcVTkQV@lsg-oss-chatgpt-agi-hcfs.oss-ap-southeast-1-internal.aliyuncs.com/crawl/multimodal/HuggingFaceFW/fineweb-edu/sample/100BT/ /prodcpfs/user/fengmingquan/dataset/raw/fineweb-edu/sample/100BT-25BT/

juicefs sync -u -p 100 oss://LTAI5tDuCoTTh6gu5PK8gFfN:6eZGIEeLo81eRSqYRMHUEG2FcVTkQV@lsg-oss-chatgpt-agi-hcfs.oss-ap-southeast-1-internal.aliyuncs.com/crawl/multimodal/allenai/openbookqa /prodcpfs/user/fengmingquan/dataset/raw/openbookqa

#git config --global user.email "fengmingquan@sjtu.edu.cn"
#git config --global user.name "fengmingquan"

ssh root@dsw-notebook-dsw-8eldzhg618bk32qyse-12534.vpc-t4nptn1gzsxxk04su5mpc.instance-forward.dsw.ap-southeast-1.aliyuncs.com -p 22


nohup /cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python test_infi.py > log/test_infi_0.log 2>&1 &


/cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/torchrun --standalone --nproc_per_node=8 /cpfs/user/fengmingquan/nanoGPT/infer_topk.py \
  --model /cpfs/user/fengmingquan/nanoGPT/out-prodcpfs/qwen2-0.5B-red/2025-10-11_14-07-44 \
  --dataset /prodcpfs/user/fengmingquan/dataset/processed-qwen2 \
  --dataset_prefix "red-1.0bt" \
  --split train \
  --block_size 4096 \
  --top_k 50 \
  --out_dir /cpfs/user/fengmingquan/nanoGPT/infer_out