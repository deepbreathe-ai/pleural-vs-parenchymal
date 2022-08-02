import yaml
import os
import pandas as pd
from tqdm import tqdm
import wget

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

def data_pull(fold=None):
    '''
    Pull raw clips from AWS database using a generated query in csv format
    '''
    if fold is not None:
        output_folder = cfg['PATHS']['RAW_CLIPS_DIR'] + 'fold_{}/'.format(fold)
    else:
        output_folder = cfg['PATHS']['RAW_CLIPS_DIR']

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    if fold is not None:
        df = pd.read_csv('data/fold_' + str(fold) + '_clips_table.csv')
    else:
        df = pd.read_csv(cfg['PATHS']['CLIPS_TABLE'])

    print('Getting AWS links...')

    # Dataframe of all clip links
    links = df.s3_path

    print('Fetching clips from AWS...')

    # Download clips and save to disk
    for link in tqdm(links):
        print(link)
        firstpos = link.rfind("/")
        lastpos = len(link)
        filename = link[firstpos+1:lastpos]
        if "VID" not in filename:
            print(filename)
        wget.download(link, output_folder + filename)

    print('Fetched clips successfully!')

    return

if __name__ == '__main__':
    data_pull()

    # for fold in range(10):
    #     data_pull(fold=fold)
