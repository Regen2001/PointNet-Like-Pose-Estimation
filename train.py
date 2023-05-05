import os
from tqdm import tqdm

command = ['python train_classification.py',
           'python train_sign.py',
           'python train_rotation.py',
           'python train_rotation.py --log_dir rotation_l1_sum --loss_function l1_loss --loss_reduction sum',
           'python train_rotation.py --log_dir rotation_l1_mean --loss_function l1_loss --loss_reduction mean',
           'python train_rotation.py --log_dir rotation_l2_sum --loss_function l2_loss --loss_reduction sum',
           'python train_rotation.py --log_dir rotation --loss_function l2_loss --loss_reduction mean',
           'python train_translation.py --log_dir translation_l1_sum --loss_function l1_loss --loss_reduction sum',
           'python train_translation.py --log_dir translation_l1_mean --loss_function l1_loss --loss_reduction mean',
           'python train_translation.py --log_dir translation_l2_sum --loss_function l2_loss --loss_reduction sum',
           'python train_translation.py --log_dir translation',
           'python train_translation.py --log_dir translation_l1_sum_no_mlp --loss_function l1_loss --loss_reduction sum --use_mean_mlp False',
           'python train_translation.py --log_dir translation_l1_mean_no_mlp --loss_function l1_loss --loss_reduction mean --use_mean_mlp False',
           'python train_translation.py --log_dir translation_l2_sum_no_mlp --loss_function l2_loss --loss_reduction sum --use_mean_mlp False',
           'python train_translation.py --log_dir translation_no_mlp --use_mean_mlp False']

test_command = ['python test_clsaaification.py',
                'python test_sign.py',
                'python test_translation.py --log_dir translation_l1_sum',
                'python test_translation.py --log_dir translation_l1_mean',
                'python test_translation.py --log_dir translation_l2_sum',
                'python test_translation.py --log_dir translation',
                'python test_translation.py --log_dir translation_l1_sum_no_mlp --use_mean_mlp False',
                'python test_translation.py --log_dir translation_l1_mean_no_mlp --use_mean_mlp False',
                'python test_translation.py --log_dir translation_l2_sum_no_mlp --use_mean_mlp False',
                'python test_translation.py --log_dir translation_no_mlp --use_mean_mlp False',
                'python test_rotation.py --log_dir rotation_l1_sum',
                'python test_rotation.py --log_dir rotation_l1_mean',
                'python test_rotation.py --log_dir rotation_l2_sum',
                'python test_rotation.py --log_dir rotation']

i = 0
for com in tqdm(command):
    os.system(com)
    os.system(test_command[i])
    os.system('cls')