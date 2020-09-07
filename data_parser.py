#---------------------------------
#		NAME			||	AM	||
#	Georgios Vardakas	||	432	||
#   Iro Spyrou          ||  440 ||
#---------------------------------
#	Biomedical Data Analysis
# 	Written in Python 3.6

import scipy.io
import os
import csv
import heartpy as hp
import numpy as np

class Data_Parser:
    def __init__(self, data_path, sample_rate):
        self.sample_rate = sample_rate
        self.data_path = data_path

    def read_data(self, path):
        all_files = os.listdir(path)
        trainSet_dict, labels_dict = dict(), dict()
        trainingSet, labels = list(), list()

        for felement in all_files:
            full_path = path + felement
            if ".mat" in felement:
                mat_data = scipy.io.loadmat(full_path)
                file_name = full_path.split('/')[2]
                file_name = file_name.split('.')[0]
                trainSet_dict[file_name] = mat_data['val'][0]

            if "REFERENCE.csv" == felement:
                with open(full_path, newline='') as csvfile:
                    linereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    for index, line in enumerate(linereader):
                        line = line[0].split(',')
                        file_name = line[0]
                        label = line[1]
                        labels_dict[file_name] = label

        for file_key in trainSet_dict:
            trainingSet.append(trainSet_dict[file_key])
            labels.append(labels_dict[file_key])

        return np.asarray(trainingSet), np.asarray(labels)

    def create_datasets(self, trainingSet, labels):
        normal_dataset, normal_labels = list(), list()
        atrfib_dataset, atrfib_labels = list(), list()
        other_dataset, other_labels = list(), list()

        for index, label in enumerate(labels):
            if label == 'A':
                atrfib_dataset.append(trainingSet[index])
                atrfib_labels.append(1)
            if label == 'N':
                normal_dataset.append(trainingSet[index])
                normal_labels.append(0)
            if label == 'O':
                other_dataset.append(trainingSet[index])
                other_labels.append(0)

        atrfib_dataset = np.asarray(atrfib_dataset)
        atrfib_labels = np.asarray(atrfib_labels)
        normal_dataset = np.asarray(normal_dataset)
        normal_labels = np.asarray(normal_labels)
        other_dataset = np.asarray(other_dataset)
        other_labels = np.asarray(other_labels)

        return atrfib_dataset, atrfib_labels, normal_dataset, normal_labels, other_dataset, other_labels

    def preprocess_dataset(self, dataset):
        preprocessed_dataset = list()
        for datapoint in dataset:
            datapoint = hp.preprocessing.scale_data(datapoint)
            datapoint = hp.preprocessing.enhance_ecg_peaks(datapoint, sample_rate = self.sample_rate, iterations = 3)
            preprocessed_dataset.append(datapoint)

        return np.asarray(preprocessed_dataset)

    def read_preprocessed_dataset(self):
        if not os.path.exists("Preprocessed_data"):
            os.makedirs("Preprocessed_data")
            data, labels = self.read_data(self.data_path)
            atrfib_dataset, atrfib_labels, normal_dataset, normal_labels, other_dataset, other_labels = self.create_datasets(data, labels)
            atrfib_dataset = self.preprocess_dataset(atrfib_dataset)
            normal_dataset = self.preprocess_dataset(normal_dataset)
            other_dataset = self.preprocess_dataset(other_dataset)

            np.savez("Preprocessed_data/atrfib_dataset.npz", *atrfib_dataset)
            np.savez("Preprocessed_data/normal_dataset.npz", *normal_dataset)
            np.savez("Preprocessed_data/other_dataset.npz", *other_dataset)

            np.savez("Preprocessed_data/atrfib_labels.npz", *atrfib_labels)
            np.savez("Preprocessed_data/normal_labels.npz", *normal_labels)
            np.savez("Preprocessed_data/other_labels.npz", *other_labels)

            return atrfib_dataset, atrfib_labels, normal_dataset, normal_labels, other_dataset, other_labels
        else:
            container = np.load("Preprocessed_data/atrfib_dataset.npz")
            atrfib_dataset = [container[key] for key in container]
            atrfib_dataset = np.asarray(atrfib_dataset)

            container = np.load("Preprocessed_data/normal_dataset.npz")
            normal_dataset = [container[key] for key in container]
            normal_dataset = np.asarray(normal_dataset)

            container = np.load("Preprocessed_data/other_dataset.npz")
            other_dataset = [container[key] for key in container]
            other_dataset = np.asarray(other_dataset)

            container = np.load("Preprocessed_data/atrfib_labels.npz")
            atrfib_labels = [container[key] for key in container]
            atrfib_labels = np.asarray(atrfib_labels)

            container = np.load("Preprocessed_data/normal_labels.npz")
            normal_labels = [container[key] for key in container]
            normal_labels = np.asarray(normal_labels)

            container = np.load("Preprocessed_data/other_labels.npz")
            other_labels = [container[key] for key in container]
            other_labels = np.asarray(other_labels)

            return atrfib_dataset, atrfib_labels, normal_dataset, normal_labels, other_dataset, other_labels
