#!/bin/bash
#SBATCH --job-name=unet16_4                 # Название задачи 
#SBATCH --error=unet16_4-%j.err             # Файл для вывода ошибок 
#SBATCH --output=unet16_4-%j.log            # Файл для вывода результатов
#SBATCH --time=24:00:00 
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу 
#SBATCH --gpus=1

/home/adkugusheva/anaconda3/bin/python /home/adkugusheva/unet_4/Unet16_main.py  