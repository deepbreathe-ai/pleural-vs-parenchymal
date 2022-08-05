import yaml
import os
import pandas as pd

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

database_query = cfg['PATHS']['DATABASE_QUERY']

mask_clip_path = cfg['PATHS']['MASKED_CLIPS_DIR']


def create_frame_dataframe(database_query):
    '''
    Extracts out pertinent information from database query csv and builds a dataframe linking filenames, patients, and class
    :database_filename: filepath to database query csv
    '''
    df = pd.read_csv(database_query)

    # Removes clips with unlabelled parenchymal findings
    df = df[df.view.notnull()]

    # Remove DNU
    df = df[df.do_not_use == 0]

    df['exam_id'] = df['exam_id'].astype(str)
    df['patient_id'] = df['patient_id'].astype(str)
    df['vid_id'] = df['vid_id'].astype(str)

    # Create filename
    df['filename'] = df['exam_id'] + "_" + df['patient_id'] + "_VID" + df['vid_id']

    for i, name in enumerate(df['filename']):
        if 'nan' in name:
            df['filename'][i] = "vid{}_vid{}_VID{}".format(i, i, i)
            print(i, name, df['filename'][i])

    print(df['view'])

    # Create column of class category to each clip.
    # Modifiable for binary or multi-class labelling
    df['class'] = df.apply(lambda row: 0 if row['view'] == 'parenchymal' else (1 if row['view'] == 'pleural' else -1),
                           axis=1)

    df['Path'] = df.apply(lambda row: row.filename, axis=1)

    df['s3_path'] = df.apply(lambda row: row.s3_path, axis=1)

    df.to_csv(cfg['PATHS']['CLIPS_TABLE'], index=False)

    return df

def create_fold_dataframe(clips_table_path,fold):
    '''
    Creates clips table for each cross-validation fold from the original clips table, and the partitions saved during training.
    :clips_table_path: path to clips table of entire training set
    :fold: cross-validation fold (0-9) of the clips table you'd like to build
    '''
    fold_df = pd.read_csv(cfg['PATHS']['PARTITIONS_DIR'] + 'fold_{}_val_set.csv'.format(fold))
    clips_df = pd.read_csv(clips_table_path)
    files = list(fold_df['Clip'].astype(str).unique())
    fold_table_df = clips_df.loc[clips_df['filename'].isin(files)]
    fold_table_df.to_csv('data/fold_' + str(fold) + '_clips_table.csv',index=False)
    return fold_table_df


if __name__ == "__main__":
    create_frame_dataframe(database_query)

    # for fold in range(10):
    #     create_fold_dataframe('data/train_query.csv',fold)