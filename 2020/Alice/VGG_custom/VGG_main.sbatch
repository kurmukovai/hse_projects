#!/bin/bash
#SBATCH --job-name=exp_26                 # Название задачи 
#SBATCH --error=exp_26-%j.err             # Файл для вывода ошибок 
#SBATCH --output=exp_26-%j.log            # Файл для вывода результатов
#SBATCH --time=24:00:00 
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу 
#SBATCH --gpus=1

/home/adkugusheva/anaconda3/bin/python /home/adkugusheva/exp_26/VGG_main.py  