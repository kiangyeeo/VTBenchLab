import pandas as pd
import argparse

def add_image_paths(input_csv_file):
    images_per_folder=10000

    print('Reading CSV file:', input_csv_file)
    df = pd.read_csv(input_csv_file)
    df.rename(columns={'Image Path': 'Image Url'}, inplace=True)
    print('Total images:', len(df))
    total_images = len(df)
    dataset_name = input_csv_file.split('_')[0]
    image_path = [ f"{dataset_name}/images/{folder:07d}/{image_index:07d}.jpg"  for folder in range(total_images//images_per_folder+1) for image_index in range(images_per_folder) ][:total_images]
    df['Image Path'] = image_path
    # save
    df.to_csv(input_csv_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add image paths to CSV file')
    parser.add_argument('input_csv_file', type=str, help='Input CSV file path')
    
    args = parser.parse_args()
    add_image_paths(args.input_csv_file)