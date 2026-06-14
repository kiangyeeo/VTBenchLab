import argparse
import pandas as pd
import numpy as np
import requests
import zlib
import os
import shelve
import magic
from multiprocessing import Pool
from tqdm import tqdm

# Parse command line arguments
headers = {
    'User-Agent':'Googlebot-Image/1.0', # Pretend to be googlebot
    'X-Forwarded-For': '64.18.15.200'
}

def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    return (split_ind, r)

def df_multiprocess(df, processes, chunk_size, func, dataset_name):
    print("Generating parts...")
    with shelve.open('%s_%s_%s_results.tmp' % (dataset_name, func.__name__, chunk_size)) as results:

        pbar = tqdm(total=len(df), position=0)
        # Resume:
        finished_chunks = set([int(k) for k in results.keys()])
        pbar.desc = "Resuming"
        for k in results.keys():
            pbar.update(len(results[str(k)][1]))

        pool_data = ((index, df[i:i + chunk_size], func) for index, i in enumerate(range(0, len(df), chunk_size)) if index not in finished_chunks)
        print(int(len(df) / chunk_size), "parts.", chunk_size, "per part.", "Using", processes, "processes")

        pbar.desc = "Downloading"
        with Pool(processes) as pool:
            for i, result in enumerate(pool.imap_unordered(_df_split_apply, pool_data, 2)):
                results[str(result[0])] = result
                pbar.update(len(result[1]))
        pbar.close()

    print("Finished Downloading.")
    return


# For checking mimetypes separately without download
def check_mimetype(row):
    if os.path.isfile(str(row['file'])):
        row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
    return row

# Don't download image, just check with a HEAD request, can't resume.
# Can use this instead of download_image to get HTTP status codes.
def check_download(row):
    fname = _file_name(row)
    try:
        # not all sites will support HEAD
        response = requests.head(row['url'], stream=False, timeout=5, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        row['headers'] = dict(response.headers)
    except:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row
    if response.ok:
        row['file'] = fname
    return row

def download_image(row):
    fname = row['Image Path']
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    # Skip Already downloaded, retry others later
    if os.path.isfile(fname):
        row['status'] = 200
        row['file'] = fname
        row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
        return row

    try:
        # use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(row['url'], stream=False, timeout=10, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        #row['headers'] = dict(response.headers)
    except Exception as e:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row

    if response.ok:
        try:
            with open(fname, 'wb') as out_file:
                # some sites respond with gzip transport encoding
                response.raw.decode_content = True
                out_file.write(response.content)
            row['mimetype'] = magic.from_file(fname, mime=True)
            row['size'] = os.stat(fname).st_size
        except:
            # This is if it times out during a download or decode
            row['status'] = 408
            return row
        row['file'] = fname
    return row

def open_csv(fname, folder, start_index=None, end_index=None):
    print("Opening data file %s..." % fname)
    df = pd.read_csv(fname, usecols=['Image Url', 'Image Path'])
    df = df.rename(columns={'Image Url': 'url'})
    df['folder'] = folder
    if start_index is not None and end_index is not None:
        print("Slicing dataframe from %d to %d" % (start_index, end_index))
        df = df.iloc[start_index:end_index]
    print("Processing", len(df), "images:")
    return df

def df_from_shelve(chunk_size, func, dataset_name):
    print("Generating Dataframe from results...")
    with shelve.open('%s_%s_%s_results.tmp' % (dataset_name, func.__name__, chunk_size)) as results:
        keylist = sorted([int(k) for k in results.keys()])
        df = pd.concat([results[str(k)][1] for k in keylist], sort=True)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download images from a CSV file containing URLs')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing image URLs')
    parser.add_argument('--start_index', type=int, default=0, help='Start index for processing the CSV file')
    parser.add_argument('--end_index', type=int, default=None, help='End index for processing the CSV file') 
    parser.add_argument('--num_processes', type=int, default=64, help='Number of parallel processes to use')
    parser.add_argument('--chunk_size', type=int, default=500, help='Number of images per chunk per process')
    parser.add_argument('--data_name', type=str, default='training', help='Name of the dataset')
    args = parser.parse_args()


    df = open_csv(args.csv_path, args.data_name, args.start_index, args.end_index)
    df_multiprocess(df=df, processes=args.num_processes, chunk_size=args.chunk_size, func=download_image, dataset_name=args.data_name)
    df = df_from_shelve(chunk_size=args.chunk_size, func=download_image, dataset_name=args.data_name)
    df.to_csv("downloaded_%s_report.csv" % args.data_name, index=False)
    print("Saved.")
