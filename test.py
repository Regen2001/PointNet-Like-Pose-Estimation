import os

os.system('python test_clsaaification.py')

os.system('python test_sign.py')

os.system('python test_translation.py --log_dir translation_l1_sum')
os.system('python test_translation.py --log_dir translation_l1_mean')
os.system('python test_translation.py --log_dir translation_l2_sum')
os.system('python test_translation.py --log_dir translation')

os.system('python test_translation.py --log_dir translation_l1_sum_no_mlp --use_mean_mlp False')
os.system('python test_translation.py --log_dir translation_l1_mean_no_mlp --use_mean_mlp False')
os.system('python test_translation.py --log_dir translation_l2_sum_no_mlp --use_mean_mlp False')
os.system('python test_translation.py --log_dir translation_no_mlp --use_mean_mlp False')

os.system('python test_rotation.py --log_dir rotation_l1_sum')
os.system('python test_rotation.py --log_dir rotation_l1_mean')
os.system('python test_rotation.py --log_dir rotation_l2_sum')
os.system('python test_rotation.py --log_dir rotation')