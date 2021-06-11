#!/bin/bash
#SBATCH --job-name=unet16_5                 # Название задачи 
#SBATCH --error=unet16_5-%j.err             # Файл для вывода ошибок 
#SBATCH --output=unet16_5-%j.log            # Файл для вывода результатов
#SBATCH --time=24:00:00 
#SBATCH --cpus-per-task=1                   # Количество CPU на одну задачу 
#SBATCH --gpus=1

/home/adkugusheva/anaconda3/bin/python /home/adkugusheva/unet_5/Unet16_main.py  