from datasets.nirscene1_dataset import build_nir
from datasets.src_nirscene1_dataset import build_src_nir
from datasets.SE_sar_opt_dataset import build_SE
from datasets.src_SE_sar_opt_dataset import build_src_SE

def build_dataset(args):
    if args.data_name == 'NIR':
        train_data_file="/home/TopKWindows/Femit/nirscene1/train.txt"
        test_data_file="/home/TopKWindows/Femit/nirscene1/448x448_test.txt"

        return build_nir(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'src_nir':
        train_data_file="/home/TopKWindows/Femit/nirscene1/train.txt"
        test_data_file="/home/TopKWindows/Femit/nirscene1/448x448_test.txt"
        return build_src_nir(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=8
                )
    if args.data_name == 'SE':
        train_data_file="/home/ly/data/dataset/train.txt"
        test_data_file="/home/ly/data/dataset/train.txt"
        return build_SE(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=16
                )
    if args.data_name == 'src_SE':
        train_data_file="/home/ly/data/dataset/se_train.txt"
        test_data_file="/home/ly/data/dataset/se_winter_train.txt"
        return build_src_SE(
                train_data_file=train_data_file,
                test_data_file=test_data_file,
                size=(320, 320),
                stride=16
                )
