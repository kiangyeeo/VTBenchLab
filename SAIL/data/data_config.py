import os
SCRATCH = os.environ['SCRATCH']

DATADIR = {

    'dreamclipcc3m': {
        'annotation':f'{SCRATCH}/datasets/DownloadCC3M/cc3m_3long_1raw_captions_filtered.csv',
        'imagedir':f'{SCRATCH}/datasets/DownloadCC3M'
        },
    'dreamclipcc12m': {
        'annotation':f'{SCRATCH}/datasets/DownloadCC3M/cc12m_3long_3short_1raw_captions_url_filtered.csv',
        'imagedir':f'{SCRATCH}/datasets/DownloadCC3M'
        },
    'yfcc15m': {
        'annotation':f'{SCRATCH}/datasets/DownloadCC3M/yfcc15m_3long_3short_1raw_captions_url_filtered.csv',
        'imagedir':f'{SCRATCH}/datasets/DownloadCC3M'
        },
    
}