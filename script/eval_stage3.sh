ckpt_path=out/stage3_finetuning/best_checkpoint.pth

# single gpu inference
# RGB-pose setting
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
   --batch-size 1 \
   --gradient-accumulation-steps 1 \
   --epochs 1 \
   --opt AdamW \
   --lr 3e-4 \
   --output_dir out/test \
   --finetune $ckpt_path \
   --dataset CSL_News \
   --task SLT \
   --eval \
   --rgb_support

# # pose-only setting
#deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
#   --batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --epochs 20 \
#   --opt AdamW \
#   --lr 3e-4 \
#   --output_dir out/test \
#   --finetune $ckpt_path \
#   --dataset CSL_Daily \
#   --task SLT \
#   --eval \