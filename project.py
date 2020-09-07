#---------------------------------
#		NAME			||	AM	||
#	Georgios Vardakas	||	432	||
#   Iro Spyrou          ||  440 ||
#---------------------------------
#	Biomedical Data Analysis
# 	Written in Python 3.6

import sys
import os
from data_parser import Data_Parser
import heartpy as hp
import math
import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import collections

electrocardiogram_sample_rate = 300.0

def reduce_dataset(dataset, labels, number):
    data_ordering = np.random.permutation(dataset.shape[0])
    dataset = dataset[data_ordering]
    labels = labels[data_ordering]
    return dataset[ : number], labels[ : number]

#RR intervals returned in ms, ((t2 - t1) / sample rate) * 1000.0
def create_RR_intervals_and_measures(dataset, labels):
    temp_labels = list()
    RR_intervals = list()
    measures = list()
    for index, heart_signal in enumerate(dataset):
        try:
            #plot_RR_Peaks(heart_signal)
            working_data, measure = hp.process(heart_signal, sample_rate = electrocardiogram_sample_rate)
            dict_counter = collections.Counter(working_data['binary_peaklist'])
            rejected_threshold = dict_counter[0] / (dict_counter[0] + dict_counter[1])
            #unpacking the dictonary values
            measure = [*measure.values()]
            if (True in np.isnan(np.array(measure)) or rejected_threshold >= 0.15): continue
            measures.append(measure)
            RR_intervals.append(working_data['RR_list'])
            temp_labels.append(labels[index])
        except:
            #plotTimeSerie(heart_signal)
            continue

    return np.asarray(RR_intervals), np.asarray(measures), np.asarray(temp_labels)

def create_histogram(RR_intervals, number_of_bins):
    RR_histograms = list()
    for RR_inter in RR_intervals:
        histogram = np.histogram(RR_inter, number_of_bins)[0]
        RR_histograms.append(histogram)
    return np.asarray(RR_histograms)

def standardization(data):
	std_scaler = StandardScaler()
	std_data = std_scaler.fit_transform(data)
	return std_data

def plotTimeSerie(timeserie):
    plt.plot(timeserie, label='ECG')
    plt.xlabel('Time')
    plt.ylabel('Amplitute')
    plt.legend() #this is needed to show the label of any category and give separate color to it
    plt.grid(True) #enable the grid
    plt.tight_layout() #something to do with padding
    plt.show()

def plot_RR_Peaks(heart_signal):
    working_data, measures = hp.process(heart_signal, sample_rate = electrocardiogram_sample_rate)
    hp.plotter(working_data, measures)

def SVM_Classifier(train_set, train_labels, test_set, test_labels):
	svm_classifier = SVC()
	svm_classifier.fit(train_set, train_labels)
	prediction = svm_classifier.predict(test_set)
	return accuracy_score(test_labels, prediction), f1_score(test_labels, prediction)

def K_Neighbors_Classifier(train_set, train_labels, test_set, test_labels):
	kn_classifier = KNeighborsClassifier(n_neighbors = 3)
	kn_classifier.fit(train_set, train_labels)
	prediction = kn_classifier.predict(test_set)
	return accuracy_score(test_labels, prediction), f1_score(test_labels, prediction)

def GaussianNB_Classifier(train_set, train_labels, test_set, test_labels):
	gnb_classifier = GaussianNB()
	gnb_classifier.fit(train_set, train_labels)
	prediction = gnb_classifier.predict(test_set)
	return accuracy_score(test_labels, prediction), f1_score(test_labels, prediction)

def RandomForest_Classifier(train_set, train_labels, test_set, test_labels):
	rf_classifier = RandomForestClassifier()
	rf_classifier.fit(train_set, train_labels)
	prediction = rf_classifier.predict(test_set)
	return accuracy_score(test_labels, prediction), f1_score(test_labels, prediction)

def MLP_Classifier(train_set, train_labels, test_set, test_labels):
	mlp_classifier = MLPClassifier(hidden_layer_sizes = (100, 50), max_iter = 2000)
	mlp_classifier.fit(train_set, train_labels)
	predict = mlp_classifier.predict(test_set)
	return accuracy_score(test_labels, predict), f1_score(test_labels, predict)

def work(dataset, labels):
    folds = 10
    k_fold = KFold(n_splits = folds, shuffle = True)

    sum_accuracy_svm, sum_f1_svm = 0.0, 0.0
    sum_accuracy_knn, sum_f1_knn = 0.0, 0.0
    sum_accuracy_gnb, sum_f1_gnb = 0.0, 0.0
    sum_accuracy_rf, sum_f1_rf = 0.0, 0.0
    sum_accuracy_mlp, sum_f1_mlp = 0.0, 0.0
    for train_index, test_index in k_fold.split(dataset):
        accuracy_svm_i, f1_svm_i = SVM_Classifier(dataset[train_index], labels[train_index], dataset[test_index], labels[test_index])
        sum_accuracy_svm += accuracy_svm_i
        sum_f1_svm += f1_svm_i

        accuracy_knn_i, f1_knn_i = K_Neighbors_Classifier(dataset[train_index], labels[train_index], dataset[test_index], labels[test_index])
        sum_accuracy_knn += accuracy_knn_i
        sum_f1_knn += f1_knn_i

        accuracy_gnb_i, f1_gnb_i = GaussianNB_Classifier(dataset[train_index], labels[train_index], dataset[test_index], labels[test_index])
        sum_accuracy_gnb += accuracy_gnb_i
        sum_f1_gnb += f1_gnb_i

        accuracy_rf_i, f1_rf_i = RandomForest_Classifier(dataset[train_index], labels[train_index], dataset[test_index], labels[test_index])
        sum_accuracy_rf += accuracy_rf_i
        sum_f1_rf += f1_rf_i

        accuracy_mlp_i, f1_mlp_i = MLP_Classifier(dataset[train_index], labels[train_index], dataset[test_index], labels[test_index])
        sum_accuracy_mlp += accuracy_mlp_i
        sum_f1_mlp += f1_mlp_i

    print("")
    print("Classification Results.")
    print("Support vector machine classifier: Accuracy = {:.2f}, F1 score = {:.2f}".format((sum_accuracy_svm / folds), (sum_f1_svm / folds)))
    print("K-nearest neighbor classifier: Accuracy = {:.2f}, F1 score = {:.2f}".format((sum_accuracy_knn / folds), (sum_f1_knn / folds)))
    print("Naive Bayes classifier: Accuracy = {:.2f}, F1 score = {:.2f}".format((sum_accuracy_gnb / folds), (sum_f1_gnb / folds)))
    print("Random forest classifier: Accuracy = {:.2f}, F1 score = {:.2f}".format((sum_accuracy_rf / folds), (sum_f1_rf / folds)))
    print("Multilayer perceptron classifier: Accuracy = {:.2f}, F1 score = {:.2f}".format((sum_accuracy_mlp / folds), (sum_f1_mlp / folds)))
    print("")

def Atrial_Fibrillation_Or_Normal(atrfib_dataset, atrfib_labels, normal_dataset, normal_labels):
    normal_dataset, normal_labels = reduce_dataset(normal_dataset, normal_labels, atrfib_dataset.shape[0])

    atrfib_RR_intervals, artfib_measures, atrfib_labels = create_RR_intervals_and_measures(atrfib_dataset, atrfib_labels)
    normal_RR_intervals, normal_measures, normal_labels = create_RR_intervals_and_measures(normal_dataset, normal_labels)

    #atrfib_RR_histogram = create_histogram(atrfib_RR_intervals, number_of_bins = 5)
    #normal_RR_histogram = create_histogram(normal_RR_intervals, number_of_bins = 5)

    atrfib_normal_dataset = np.row_stack((artfib_measures, normal_measures))
    atrfib_normal_labels = np.asarray([*atrfib_labels, *normal_labels])

    atrfib_normal_dataset = standardization(atrfib_normal_dataset)

    work(atrfib_normal_dataset, atrfib_normal_labels)

def Atrial_Fibrillation_Or_Other(atrfib_dataset, atrfib_labels, other_dataset, other_labels):
    other_dataset, other_labels = reduce_dataset(other_dataset, other_labels, atrfib_dataset.shape[0])

    atrfib_RR_intervals, artfib_measures, atrfib_labels = create_RR_intervals_and_measures(atrfib_dataset, atrfib_labels)
    other_RR_intervals, other_measures, other_labels = create_RR_intervals_and_measures(other_dataset, other_labels)

    #atrfib_RR_histogram = create_histogram(atrfib_RR_intervals, number_of_bins = 5)
    #other_RR_histogram = create_histogram(other_RR_intervals, number_of_bins = 5)

    atrfib_other_dataset = np.row_stack((artfib_measures, other_measures))
    atrfib_other_labels = np.asarray([*atrfib_labels, *other_labels])

    atrfib_other_dataset = standardization(atrfib_other_dataset)

    work(atrfib_other_dataset, atrfib_other_labels)

def main(argv):
    data_path = "./training2017/"
    my_data_parser = Data_Parser(data_path, electrocardiogram_sample_rate)
    atrfib_dataset, atrfib_labels, normal_dataset, normal_labels, other_dataset, other_labels = my_data_parser.read_preprocessed_dataset()
    Atrial_Fibrillation_Or_Normal(atrfib_dataset, atrfib_labels, normal_dataset, normal_labels)
    Atrial_Fibrillation_Or_Other(atrfib_dataset, atrfib_labels, other_dataset, other_labels)


if __name__ == "__main__":
	main(sys.argv[1:])
