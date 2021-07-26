import os
import shutil
import tensorflow

#%%

#Sort tfRecords by country and year and move into Folders

#%%

#Code jd - adapted from  split_surveys.py
def create_folders(in_file, out_dir):
    '''
    Args
    - inFile: str, path to tf_record file
    - out_dir: str, path to output directory
    '''
    tensorflow.enable_eager_execution() #only used in tensorflow 1 
    os.makedirs(out_dir, exist_ok=True)
    #load and read ExampleMessage
    data = tensorflow.data.TFRecordDataset(in_file)
    for raw_record in data.take(1):
        example = tensorflow.train.Example()
        example.ParseFromString(raw_record.numpy())

        #get original strings back from features
        survey_country = example.features.feature['country'].bytes_list.value[0].decode()
        country_str = survey_country.lower().replace(' ', '_').replace("'", '_') #modify country-string to make them the same shape (replace whitespace)
        year = example.features.feature['year'].int64_list.value[0]
        #build paths out of feature values: country and year
        file_code = f'{country_str}_{year}'
        file_out_dir = os.path.join(out_dir, file_code)
        concat_name =  os.path.join(file_out_dir, tf_file) #concatenate year and country
        #create directories for country and year e.g. angola_2015
        if not os.path.exists(file_out_dir):
            os.makedirs(file_out_dir)
        if not os.path.exists(concat_name):
            shutil.move(in_file, concat_name) #copying files on my maschine. Might change to move(mv) on Colab
#%%
root_path = '/mnt/datadisk/sciebo/DIS22/Data_Acquisition/s2tfrec'
tfrecord_list = os.listdir(root_path) #get list of all tfrecords
out_dir = '/home/stoermer/01_data'
#execute function for every file
for tf_file in tfrecord_list:
    file_name = os.path.join(root_path, tf_file)
    create_folders(in_file=file_name, out_dir=out_dir)
