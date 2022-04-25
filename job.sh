#!/bin/bash
#SBATCH --job-name=cvt   ## job name
#SBATCH --nodes=1                ## 索取 1 節點
#SBATCH --gres=gpu:1             ## 每個節點索取 8 GPUs
##### SBATCH --time=72:00:00          ## 最長跑 10 分鐘 (測試完這邊記得改掉，或是直接刪除該行)
#SBATCH --account=MST110018     ## PROJECT_ID 請填入計畫ID(ex: MST108XXX)，扣款也會根據此計畫ID
#SBATCH --partition=gp2d        ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、p4d(最長跑4天)
#SBATCH -o result.out
module purge
module load miniconda3
# module load cuda
conda activate cvt

# 大部分使用 conda 用戶，程式並沒有透過 MPI 溝通
# 應不需要再添加 srun/mpirun，直接加上需要運行的指令即可

python3 train.py --cfg config.yaml