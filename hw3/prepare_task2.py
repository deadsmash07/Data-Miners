import os
import argparse
import numpy as np
import torch

def load_train(data_dir):
    """
    Load the .npy files for the train split and convert them to torch tensors.
    Returns a dict with keys: user_features, product_features, user_product, labels
    """
    split_dir = os.path.join(data_dir, 'train')
    # Paths for npy files
    uf_path = os.path.join(split_dir, 'user_features.npy')
    pf_path = os.path.join(split_dir, 'product_features.npy')
    up_path = os.path.join(split_dir, 'user_product.npy')
    lbl_path = os.path.join(split_dir, 'label.npy')

    # Load numpy arrays
    user_features = np.load(uf_path)
    product_features = np.load(pf_path)
    user_product = np.load(up_path)
    labels = np.load(lbl_path)

    # Convert to torch tensors
    return {
        'user_features': torch.from_numpy(user_features).float(),
        'product_features': torch.from_numpy(product_features).float(),
        'user_product': torch.from_numpy(user_product).long(),
        'labels': torch.from_numpy(labels).long()
    }


def prepare_dataset(data_dir, output_file):
    """
    Prepare the dataset by loading the train split and saving it into a .pt file.
    """
    print("Loading train split from", os.path.join(data_dir, 'train'))
    train_data = load_train(data_dir)
    shapes = {k: v.shape for k, v in train_data.items()}
    print("Loaded train data shapes:", shapes)

    print(f"Saving processed data to {output_file}...")
    torch.save(train_data, output_file)
    print("Save complete.")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Task2 train data: load .npy files from train/ and save as a .pt file.'
    )
    parser.add_argument(
        '--data_dir', '-d', type=str, required=True,
        help='Directory containing train/ subfolder with .npy files.'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='train_task2_data.pt',
        help='Output .pt filename (default: train_task2_data.pt)'
    )

    args = parser.parse_args()
    prepare_dataset(args.data_dir, args.output)


if __name__ == '__main__':
    main()

# python3 prepare_task2.py --data_dir datasets/task2 --output task2_data.pt