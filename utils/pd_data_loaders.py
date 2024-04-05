import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

class CelebA():

    def __init__(self, main_data_folder, images_folder):
        """ get partition, attributes and return apndas in which there is both columns
            images folder wrt the script where you are running it, not wrt main_data_foler
        """
        self.main_data_folder = main_data_folder
        self.images_folder = images_folder+os.sep

        ## import the data set that include the attribute for each picture
        df_attr = pd.read_csv(main_data_folder + 'list_attr_celeba.csv')
        df_attr.set_index('image_id', inplace=True)
        df_attr.replace(to_replace=-1, value=0, inplace=True) #replace -1 by 0
        #get partition of each sample
        df_partition = pd.read_csv(main_data_folder + 'list_eval_partition.csv')
        df_partition.set_index('image_id', inplace=True)
        # join the partition with the attributes and the paths
        df_par_attr = df_partition.join(df_attr['Male'], how='inner')
        df_par_attr['path']=df_par_attr.index
        df_par_attr["path"] = df_par_attr["path"].apply(lambda x: self.images_folder+x)

        #Split into train, test, validation and get adta loaders
        self.df_train = df_par_attr[df_par_attr["partition"]==0]
        self.df_validation = df_par_attr[df_par_attr["partition"]==1]
        self.df_test = df_par_attr[df_par_attr["partition"]==2]

    def load_dataset(self):
        self.df_dataset = pd.concat([self.df_train, self.df_validation, self.df_test], axis=0)
        return self.df_dataset

    def get_train(self):
        return self.df_train

    def get_validation(self):
        return self.df_validation

    def get_test(self):
        return self.df_test


class LFWA():
    def __init__(self, main_data_folder, images_folder):
        """
        Load LFWA into dataframes with labels = [path, partition, gender]
        Args:
            main_data_folder (str): path to main_data_folder. The content must be_
                - '[female|male]_names.txt' file with list of names of males and females
                - 'peopleDev[Train|Test].txt' file with list of partition of each person
                - images_folder: (usually: 'lfw_funneled') folder with folder of people as it is given in the official repo.                
            images_folder (str): path to imager wrt main_data_folder
        """
        np.random.seed(42)
        self.main_data_folder = main_data_folder
        self.images_folder = main_data_folder+os.sep+images_folder+os.sep

        ## Prepare dataset: labels and partitions infered from folders and files
        self.ls_fem = pd.read_csv(main_data_folder+'female_names.txt', header = None).values
        self.ls_male =  pd.read_csv(main_data_folder+'male_names.txt', header = None).values

        self.ls_train = pd.read_csv(main_data_folder+'peopleDevTrain.txt', delimiter='\t').index.values
        self.ls_val = pd.read_csv(main_data_folder+'peopleDevTest.txt', delimiter='\t').index.values

        self.df_dataset = None

    def load_dataset(self):
        """
        Load LFWA into dataframes with columns = [path, partition, gender]
        Load LFWA dataset parsing gender and partition data from each file and creating each path for each picture
        Remind that each LFWA pictures are inside a folder with the name of the person
        Male:1 - Female:0

        """
        images_dict = {}
        people_folders = [f for f in os.walk(self.images_folder) if not f[0] is self.images_folder] #remove parent floder from list
        for person_folder in people_folders:
            path, _, pictures = person_folder
            person_name = path.split(os.sep)[-1]
            
            #get partition data - Person-level
            is_train = person_name in self.ls_train
            is_val = person_name in self.ls_val
            if (is_train or is_val) and not(is_train and is_val):
                partition = 0 if is_train else 1
            else:
                raise Exception(f"""WARNING: {person_name} is not in the Partition lists or in both: {is_train}, {is_val}""")

            for pic in pictures:
                pic_path = path+os.sep+pic
                images_dict[pic_path] = {}
                images_dict[pic_path]['partition'] = partition
                is_fem = pic in self.ls_fem
                is_male = pic in self.ls_male
                # Get gender data
                if is_fem or is_male and not (is_fem and is_male):
                    images_dict[pic_path]['Male'] = 1 if is_male else 0
                else:
                    raise Exception(F"""WARNING: {pic} is not in the GENDER lists or in both: {is_fem}, {is_male}""")

        self.df_dataset = pd.DataFrame.from_dict(images_dict, orient='index')
        self.df_dataset['path'] = self.df_dataset.index

        self.df_train = self.df_dataset[self.df_dataset["partition"]==0]
        self.df_val = self.df_dataset[self.df_dataset["partition"]==1]

        return self.df_dataset

    def get_train(self):
        """Return train dataset: rows where idx of partition is 0

        Returns:
            pd.DataFrame: dataframe of train images with path and gender.
        """
        return self.df_dataset[self.df_dataset["partition"]==0]

    def get_validation(self):
        """Return validation dataset: rows where idx of partition is 1

        Returns:
            pd.DataFrame: dataframe of validation images with path and gender.
        """
        return self.df_dataset[self.df_dataset["partition"]==1]

    def get_test(self):

        raise Exception('Neither LFW+A dataset has originally Test Set, nor theres need of using LFWA test set in this experiment')




class FairFaces():
    def __init__(self, main_data_folder):
        """
        Create df for FairFaces datastet. 

        Args:
            main_data_folder (str): path to folder of FairFaces. Must have:
                - fairface_label_[train|val].csv with the relative path to main folder (e.g. train/1.jpg or val/1.jpg), and age, race and gender.
        """
        np.random.seed(42)
        self.main_data_folder = main_data_folder
        self.images_folder = main_data_folder

        df_attrs_trainval = pd.read_csv(main_data_folder+'fairface_label_train.csv').rename(columns={'file':'path'})
        df_attrs_trainval["path"] = df_attrs_trainval["path"].apply(lambda x: self.images_folder+x)
        df_attrs_trainval["Male"] = df_attrs_trainval["gender"].apply(lambda x: int(x=='Male'))
        #df_attrs_trainval = df_attrs_trainval.drop('service_test')
        df_attrs_trainval.index = df_attrs_trainval['path'].values
        self.df_train_val = df_attrs_trainval

        self.df_train, self.df_validation = train_test_split(df_attrs_trainval, test_size=0.2, stratify=df_attrs_trainval[[ "gender", "age", "race"]])
        self.df_train['partition'] = 0
        self.df_validation['partition'] = 1
        
        df_attr_test = pd.read_csv(main_data_folder+'fairface_label_val.csv').rename(columns={'file':'path'})
        df_attr_test["path"] = df_attr_test["path"].apply(lambda x: self.images_folder+x)
        df_attr_test["Male"] = df_attr_test["gender"].apply(lambda x: int(x=='Male'))
        #df_attr_test = df_attr_test.drop('service_test')
        df_attr_test.index = df_attr_test['path'].values
        df_attr_test['partition'] = 2
        self.df_test = df_attr_test

        self.df_dataset = None

    def load_dataset(self):
        self.df_dataset = pd.concat([self.df_train, self.df_validation, self.df_test], axis=0)
        return self.df_dataset

    def get_train(self):
        return self.df_train

    def get_validation(self):
        return self.df_validation

    def get_test(self):
        return self.df_test
