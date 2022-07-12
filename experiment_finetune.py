import os

# regular
os.system("python main.py --model_name TwinsSVT_1d --experiment_name experiment_1d_resume_B100 --ckpt_path 13meli47")  # 0
os.system("python main.py --model_name TwinsSVT_1d_group --experiment_name experiment_1d_resume_B100_group --ckpt_path y0n9fblj")  # 1
os.system("python main.py --model_name TwinsSVT_1d_LoGlo --experiment_name experiment_1d_resume_B100_LoGlo --ckpt_path 2n6mky4b")  # 2
os.system("python main.py --model_name TwinsSVT_1d_SE --experiment_name experiment_1d_resume_B100_SE --ckpt_path 36890boe")  # 3
os.system("python main.py --model_name TwinsSVT_1d_group_LoGlo --experiment_name experiment_1d_resume_B100_group_LoGlo --ckpt_path 2etoxzeg")  # 12
os.system("python main.py --model_name TwinsSVT_1d_group_SE --experiment_name experiment_1d_resume_B100_group_SE --ckpt_path 3s4jtezs")  # 13
os.system("python main.py --model_name TwinsSVT_1d_LoGlo_SE --experiment_name experiment_1d_resume_B100_LoGlo_SE --ckpt_path 3qy669bq")  # 23
os.system("python main.py --model_name TwinsSVT_1d_group_LoGlo_SE --experiment_name experiment_1d_resume_B100_group_LoGlo_SE --ckpt_path 2ubi3sz2")  # 123

# mega
os.system("python main.py --model_name TwinsSVT_1d --experiment_name mega_experiment_1d_resume_B100 --split_train train valid --split_valid test --ckpt_path 1p27uhe0") #0
os.system("python main.py --model_name TwinsSVT_1d_group --experiment_name mega_experiment_1d_resume_B100_group --split_train train valid --split_valid test --ckpt_path 2tgtr9nj") #1
os.system("python main.py --model_name TwinsSVT_1d_LoGlo --experiment_name mega_experiment_1d_resume_B100_LoGlo --split_train train valid --split_valid test --ckpt_path 3qg7hpq9") #2
os.system("python main.py --model_name TwinsSVT_1d_SE --experiment_name mega_experiment_1d_resume_B100_SE --split_train train valid --split_valid test --ckpt_path 2tw5nknn")  # 3
os.system("python main.py --model_name TwinsSVT_1d_group_LoGlo --experiment_name mega_experiment_1d_resume_B100_group_LoGlo --split_train train valid --split_valid test --ckpt_path 1kjo39iw")  # 12
os.system("python main.py --model_name TwinsSVT_1d_group_SE --experiment_name mega_experiment_1d_resume_B100_group_SE --split_train train valid --split_valid test --ckpt_path 1ttlseji")  # 13
os.system("python main.py --model_name TwinsSVT_1d_LoGlo_SE --experiment_name mega_experiment_1d_resume_B100_LoGlo_SE --split_train train valid --split_valid test --ckpt_path 2joitq6d")  # 23
os.system("python main.py --model_name TwinsSVT_1d_group_LoGlo_SE --experiment_name mega_experiment_1d_resume_B100_group_LoGlo_SE --split_train train valid --split_valid test --ckpt_path 3qcuwcii")  # 123

# missing to run
