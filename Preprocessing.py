# %% [markdown]
'''
# Cleaning Data for Preprocessing a Time Series
'''
# %%
from nltk import data
import pandas as pd
import numpy as np
import glob

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
import unidecode
import re

import class_words

# %%
class Preprocess():

    def __init__(self, rootDir, word_dict):
        self.rootDir_ = rootDir
        self.class_words_dict_ = word_dict
        self.spanish_stemmer_ = SnowballStemmer('spanish')
        self.special_words_ = ['piez']
        self.stopwords_spanish_ = stopwords.words('spanish')
        self.df_ = pd.DataFrame(columns=['Id','Tipo','Tipo_2','Tipo_3','Tipo_4','Marca','Submarca','Empaque','Contenido','UnidadMedida','LocalidadGeografica','Fuente','precio','fecha'])

        self.data_ = self.import_data()
        self.add_stop_words()
        self.preprocess('descripcion')
        self.categorize()
        self.append_df()

    def import_data(self):
        '''
        Import all files in a library without subfolders
        '''
        data = {}
        path = self.rootDir_+'*.csv'
        for fname in glob.glob(path):
            data[fname.split('\\')[1].split('.csv')[0]] = pd.read_csv(fname, index_col=0)
            try:
                data.get(fname.split('\\')[1].split('.csv')[0])['fecha'] = pd.to_datetime(data.get(fname.split('\\')[1].split('.csv')[0])['fecha'], format='%d-%m-%Y')
            except KeyError:
                print('Check datetime values, as I didnt find them.')
        return data

    def add_stop_words(self):
        new_stop_words = ['s']
        self.stopwords_spanish_.extend(new_stop_words)

        return self

    def tokenize(self,data):
        '''
        Input: the complete strins
        Output: the tokenize string in a list of strings
        '''
        return word_tokenize(data)

    def remove_stopwords_punctuation(self, data):
        clean_description = []
        for word in data:
            if (word not in self.stopwords_spanish_ and
                word not in string.punctuation):
                clean_description.append(word)
        
        return clean_description

    def remove_accents(self,data):

        return [unidecode.unidecode(word) for word in data]

    def lowercasing(self, data):

        return [word.lower() for word in data]

    def stemming(self, data):

        return [self.spanish_stemmer_.stem(word) for word in data]

    def remove_duplicates(self, data):
        seen = set()
        result = []
        for item in data:
            if item not in seen:
                seen.add(item)
                result.append(item)
        
        return result

    def split_number_letter(self, data):
        result = []
        for word in data:
            match = re.match(r'([0-9]+)([a-z]+)', word, re.I)
            if match:
                for element in match.groups():
                    result.append(element)
            else:
                result.append(word)
        return result

    def remove_special_char(self, data):
        result = []
        for word in data:
            if (word not in self.special_words_):
                result.append(word)
        return result

    def preprocess(self, column_name):
        for values in self.data_.values():
            values[column_name] = values.apply(lambda row: self.tokenize(row[column_name]), axis=1)
            values[column_name] = values.apply(lambda row: self.remove_accents(row[column_name]),axis=1)
            values[column_name] = values.apply(lambda row: self.lowercasing(row[column_name]),axis=1)
            values[column_name] = values.apply(lambda row: self.split_number_letter(row[column_name]),axis=1)
            values[column_name] = values.apply(lambda row: self.remove_stopwords_punctuation(row[column_name]),axis=1)
            values[column_name] = values.apply(lambda row: self.stemming(row[column_name]),axis=1)
            values[column_name] = values.apply(lambda row: self.remove_special_char(row[column_name]),axis=1)
            values[column_name] = values.apply(lambda row: self.remove_duplicates(row[column_name]),axis=1)
        return self

    def append_df(self):
        for element in self.data_.keys():
            self.df_ = self.df_.append(self.data_.get(element), ignore_index=True)
        
        self.df_.drop(['producto','descripcion'], axis=1, inplace=True)
        
        return self

    def categorize(self):
        for base_key in self.data_.keys():
            self.data_.get(base_key).reset_index(drop=True,inplace=True)
            columns_to_add = ['Tipo','Tipo_2','Tipo_3','Tipo_4','Marca','Submarca','Empaque','Contenido','UnidadMedida']
            for i in columns_to_add:
                self.data_.get(base_key)[i] = np.nan
            self.data_.get(base_key)['Fuente'] = base_key
            for row in range(len(self.data_.get(base_key))):
                for element in self.data_.get(base_key)['descripcion'][row]:
                    if element in self.class_words_dict_.get('Tipo'):
                        self.data_.get(base_key)['Tipo'].loc[row] = element
                    if element in self.class_words_dict_.get('Tipo_2'):
                        self.data_.get(base_key)['Tipo_2'].loc[row] = element
                    if element in self.class_words_dict_.get('Tipo_3'):
                        self.data_.get(base_key)['Tipo_3'].loc[row] = element
                    if element in self.class_words_dict_.get('Tipo_4'):
                        self.data_.get(base_key)['Tipo_4'].loc[row] = element
                    if element in self.class_words_dict_.get('Marca'):
                        self.data_.get(base_key)['Marca'].loc[row] = element
                    if element in self.class_words_dict_.get('Submarca'):
                        self.data_.get(base_key)['Submarca'].loc[row] = element
                    if element in self.class_words_dict_.get('Empaque'):
                        self.data_.get(base_key)['Empaque'].loc[row] = element
                    if element in self.class_words_dict_.get('Contenido'):
                        self.data_.get(base_key)['Contenido'].loc[row] = element
                    if element in self.class_words_dict_.get('UnidadMedida'):
                        self.data_.get(base_key)['UnidadMedida'].loc[row] = element
                    
        return self

# %% 
rootDir = 'Dataset/'
clean_class = Preprocess(rootDir=rootDir, word_dict=class_words.class_words_dict)
# %%
clean_class.df_
# %%
clean_class.df_.to_csv('clean_data.csv', index=False)
# %%
# Imputar lo que falta con conocimiento general de cada producto.