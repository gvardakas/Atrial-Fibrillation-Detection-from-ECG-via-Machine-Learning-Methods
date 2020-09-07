# Atrial-Fibrillation-Detection-from-ECG-via-Machine-Learning-Methods

## Description of the project
Given a labeled dataset, whose data is electrocardiographs with four types of labels, we are asked to create an automated method to classify each electrocardiogram. The labels are normal heart rate, atrial brillation, other type of heart rate and noisy electrocardiogram. The noisy electrocardiographs were not used in this project. The data used for this project is taken from [The PhysioNet Computing in Cardiology Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/).

## How to run the project
1. Download the electrocardiographs data from [The PhysioNet Computing in Cardiology Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/).
2. Install the the libraries numpy, sklearn, heartpy by opening the terminal and executing the command $pip3 install numpy sklearn heartpy
3. In the project.py file change the path of the data_path variable.
4. Open a terminal and run the command $python project.py

## Notes
1. The first time the project will be executed, the data preprocessing will take some time to be over. The preprocessed data will be stored after that to a file named Preprocessed_Data.
2. The models are not stored in the code because the project was experimental.
