#!/bin/bash
#SBATCH --job-name=exp_25                 # Название задачи 
#SBATCH --error=exp_25-%j.err             # Файл для вывода ошибок 
#SBATCH --output=exp_25-%j.log            # Файл для вывода результатов
#SBATCH --time=24:00:00 
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу 
#SBATCH --gpus=1

/home/adkugusheva/anaconda3/bin/python /home/adkugusheva/exp_25/VGG16_main.py  