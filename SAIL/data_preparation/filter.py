import csv
import os
from tqdm import tqdm
from PIL import ImageFile, UnidentifiedImageError, Image  # Import UnidentifiedImageError
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Corrupt EXIF data")





def is_image_file_exist(base_path, row):
    """
    Check if image file exists.
    
    Args:
        base_path (str): Base path for images.
        row (list): Row data from CSV file.
    
    Returns:
        list or None: Returns row data if image exists, None otherwise.
    """
    image_path = os.path.join(base_path, row[-1].strip())
    try:
        # Try opening image to verify validity
        with Image.open(image_path) as img:
            img.verify()
            return row  # Return row data if successful
    except FileNotFoundError:
        return None  # Return None if image doesn't exist
    except (UnidentifiedImageError, OSError) as e:  # Catch OSError for truncated files
        print(f"Invalid image or read error: {image_path}, {e}")
        return None  # Return None if image is invalid or has read error

def main(input_file, output_file, image_base_path, chunk_size = 5000):
    

    try:
        # Pre-calculate number of lines (minus header)
        with open(input_file, 'r', encoding='utf-8') as infile:
            total_lines = sum(1 for _ in infile) - 1
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Read and write header row
            header = next(reader)
            writer.writerow(header)

            rows_chunk = []
            exist = 0
            non_exist = 0

            with tqdm(total=total_lines, desc="Processing rows") as pbar:
                for row in reader:
                    rows_chunk.append(row)
                    if len(rows_chunk) == chunk_size:
                        # Process current batch
                        for current_row in rows_chunk:
                            result = is_image_file_exist(image_base_path, current_row)
                            if result:
                                writer.writerow(result)
                                exist += 1
                            else:
                                non_exist += 1
                            pbar.update(1)
                        rows_chunk = []  # Clear current batch

                        # Update progress bar additional info
                        pbar.set_postfix({"exist": exist, "non_exist": non_exist}, refresh=True)

                # Process remaining rows in last incomplete batch
                if rows_chunk:
                    for current_row in rows_chunk:
                        result = is_image_file_exist(image_base_path, current_row)
                        if result:
                            writer.writerow(result)
                            exist += 1
                        else:
                            non_exist += 1
                        pbar.update(1)
                    pbar.set_postfix({"exist": exist, "non_exist": non_exist}, refresh=True)

            # Print final statistics
            print(f"\nProcessing complete: Number of existing images: {exist}, Number of missing images: {non_exist}")

    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter local corrupted images and remove corresponding rows in the CSV file')
    parser.add_argument('--input_csv_file', type=str, required=True, help='Path to the CSV file containing image URLs')
    parser.add_argument('--image_base_path', type=str, default='./', help='Path to the base directory containing images')
    parser.add_argument('--chunk_size', type=int, default=5000, help='Number of images per chunk')
    args = parser.parse_args()

    input_file = args.input_csv_file
    output_file = '%s_filtered.csv' % args.input_csv_file.split('.')[0]
    image_base_path = args.image_base_path
    main(input_file, output_file, image_base_path, args.chunk_size)