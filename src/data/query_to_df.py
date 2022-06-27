import yaml
import os
import pandas as pd

cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))

database_query = cfg['PATHS']['DATABASE_QUERY']

mask_clip_path = cfg['PATHS']['MASKED_CLIPS_DIR']


def create_ABline_dataframe(database_query):
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
    df['vid_id'] = df['vid_id'].astype('Int64').astype(str)

    # Create filename
    df['filename'] = df['exam_id'] + "_" + df['patient_id'] + "_VID" + df['vid_id']

    for i, name in enumerate(df['filename']):
        if 'nan' in name:
            df['filename'][i] = "vid{}_vid{}_VID{}".format(i, i, i)
            print(i, name, df['filename'][i])

    print(df['view'])

    # Create column of class category to each clip.
    # Modifiable for binary or multi-class labelling
    df['class'] = df.apply(lambda row: 0 if row['view'] == 'parenchymal' else
    (1 if row['view'] == 'pleural' else
     -1), axis=1)

    df['Path'] = df.apply(lambda row: row.filename, axis=1)

    df['s3_path'] = df.apply(lambda row: row.s3_path, axis=1)

    # Finalize dataframe
    # df = df[['filename'] + COLUMNS_WANTED + ['class'] + ['Path'] + ['s3_path']]

    # Save df - append this csv to the previous csv 'clips_by_patient_2.csv'
    df.to_csv(cfg['PATHS']['CLIPS_TABLE'], index=False)

    return df


# print(create_ABline_dataframe("parenchymal_clips.csv"))

if __name__ == "__main__":
    create_ABline_dataframe(database_query)