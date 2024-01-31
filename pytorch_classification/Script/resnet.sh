
torchrun --nproc_per_node=4 --nnodes 1 --node_rank 0 --master_addr='192.168.210.5' --master_port='9999' Tools/train.py \
  --cfg resnet.yaml