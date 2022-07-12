import os

# regular, w/o
# os.system("python main.py --model_name TwinsSVT_1d --experiment_name experiment_1d")  # 0
# os.system("python main.py --model_name TwinsSVT_1d_group --experiment_name experiment_1d_group")  # 1
# os.system("python main.py --model_name TwinsSVT_1d_LoGlo --experiment_name experiment_1d_LoGlo")  # 2
# os.system("python main.py --model_name TwinsSVT_1d_SE --experiment_name experiment_1d_SE")  # 3
os.system("python main.py --model_name TwinsSVT_1d_group_LoGlo --experiment_name experiment_1d_group_LoGlo")  # 12
os.system("python main.py --model_name TwinsSVT_1d_group_SE --experiment_name experiment_1d_group_SE")  # 13
os.system("python main.py --model_name TwinsSVT_1d_LoGlo_SE --experiment_name experiment_1d_LoGlo_SE")  # 23
os.system("python main.py --model_name TwinsSVT_1d_group_LoGlo_SE --experiment_name experiment_1d_group_LoGlo_SE")  # 123

# mega
# os.system("python main.py --model_name TwinsSVT_1d --experiment_name mega_experiment_1d --split_train train valid --split_valid test --ckpt_path 1p27uhe0")  # 0
# os.system("python main.py --model_name TwinsSVT_1d_group --experiment_name mega_experiment_1d_group --split_train train valid --split_valid test --ckpt_path 2tgtr9nj")  # 1
# os.system("python main.py --model_name TwinsSVT_1d_LoGlo --experiment_name mega_experiment_1d_LoGlo --split_train train valid --split_valid test --ckpt_path 3qg7hpq9")  # 2
# os.system("python main.py --model_name TwinsSVT_1d_SE --experiment_name mega_experiment_1d_SE --split_train train valid --split_valid test --ckpt_path 2tw5nknn")  # 3
os.system("python main.py --model_name TwinsSVT_1d_group_LoGlo --experiment_name mega_experiment_1d_group_LoGlo --split_train train valid --split_valid test")  # 12
os.system("python main.py --model_name TwinsSVT_1d_group_SE --experiment_name mega_experiment_1d_group_SE --split_train train valid --split_valid test")  # 13
os.system("python main.py --model_name TwinsSVT_1d_LoGlo_SE --experiment_name mega_experiment_1d_LoGlo_SE --split_train train valid --split_valid test")  # 23
os.system("python main.py --model_name TwinsSVT_1d_group_LoGlo_SE --experiment_name mega_experiment_1d_group_LoGlo_SE --split_train train valid --split_valid test")  # 123

# missing to run
