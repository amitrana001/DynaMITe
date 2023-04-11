import os
import pickle
import argparse
from offline_summary import get_statistics
def parse_args():
    parser = argparse.ArgumentParser("Evaluation summary from pickle path")
    parser.add_argument("--folder-path", default="", type=str, help="Path to the pickle file")
    # parser.add_argument("--ablation", default="", type=str, help="ablation or not")
    parser.add_argument("--ablation", action="store_true", help="perform evaluation only")
    return parser.parse_args()

def main():
    args = parse_args()

    folder_path = args.folder_path
    for f in os.listdir(folder_path):
        if 'bs32' in f:
            pickle_path = os.path.join(folder_path,f)
            with open(pickle_path, 'rb') as handle:
                summary_stats= pickle.load(handle)
            if args.ablation:
                get_statistics(summary_stats, ablation=True, save_stats_path=folder_path)
            else:
                get_statistics(summary_stats, ablation=False, save_stats_path=folder_path)


if __name__ == "__main__":
    main()
