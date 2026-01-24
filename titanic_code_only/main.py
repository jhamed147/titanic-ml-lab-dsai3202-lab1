import argparse
from src.train import train_model
from src.test import predict_test

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    args = parser.parse_args()

    if args.mode == 'train':
        train_model('data/train.csv')
    elif args.mode == 'test':
        predict_test(data_dir="data", models_dir="models", outputs_dir="outputs")

if __name__ == "__main__":
    main()