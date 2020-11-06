# %% [markdown]
'''
# Cleaning Data for Preprocessing a Time Series
'''
# %%
from nltk import data
import pandas as pd
import glob

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
import unidecode
import re

# %%
class Preprocess():

    def __init__(self, rootDir):
        self.rootDir = rootDir
        self.data_ = self.import_data()
        self.stopwords_spanish = stopwords.words('spanish')
        self.spanish_stemmer = SnowballStemmer('spanish')
        self.df = pd.DataFrame(columns=['Id','Tipo','Marca','Submarca','Empaque','Contenido','UnidadMedida','LocalidadGeografica','Fuente','Precio','Fecha'])
        

    def import_data(self):
        '''
        Import all files in a library without subfolders
        '''
        data = {}
        path = self.rootDir+'*.csv'
        for fname in glob.glob(path):
            data[fname.split('\\')[1].split('.csv')[0]] = pd.read_csv(fname, index_col=0)
            try:
                data.get(fname.split('\\')[1].split('.csv')[0])['fecha'] = pd.to_datetime(data.get(fname.split('\\')[1].split('.csv')[0])['fecha'], format='%d-%m-%Y')
            except KeyError:
                print('Check datetime values, as I didnt find them.')
        return data

    def tokenize(self,data):
        '''
        Input: the complete strins
        Output: the tokenize string in a list of strings
        '''
        return word_tokenize(data)

    def remove_stopwords_punctuation(self, data):
        clean_description = []
        for word in data:
            if (word not in self.stopwords_spanish and
                word not in string.punctuation):
                clean_description.append(word)
        
        return clean_description

    def remove_accents(self,data):

        return [unidecode.unidecode(word) for word in data]

    def lowercasing(self, data):

        return [word.lower() for word in data]

    def stemming(self, data):

        return [self.spanish_stemmer.stem(word) for word in data]

    def remove_duplicates(self, data):
        seen = set()
        result = []
        for item in data:
            if item not in seen:
                seen.add(item)
                result.append(item)
        
        return result

    def split_number_letter(self, data):

        return [re.findall(r'(\d+)(\w+?)', word)[0] for word in data]

    def preprocess(self, column_name):
        for values in self.data_.values():
            values[column_name] = values.apply(lambda row: self.tokenize(row[column_name]), axis=1)
            values[column_name] = values.apply(lambda row: self.remove_stopwords_punctuation(row[column_name]),axis=1)
            values[column_name] = values.apply(lambda row: self.remove_accents(row[column_name]),axis=1)
            values[column_name] = values.apply(lambda row: self.lowercasing(row[column_name]),axis=1)
            values[column_name] = values.apply(lambda row: self.stemming(row[column_name]),axis=1)
            values[column_name] = values.apply(lambda row: self.remove_duplicates(row[column_name]),axis=1)
        return self

# %% 
rootDir = 'Dataset/'
clean_class = Preprocess(rootDir=rootDir)
# %%
clean_class.data_.get('soriana')
# %%
clean_class.preprocess('descripcion')
clean_class.data_.get('soriana')
# %%
