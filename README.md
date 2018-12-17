# breast_cancer_classification
classification of breast cancer images
Description
The purpose of this mini-project is to classify image patches from histopathological samples as either containing or not containing metastatic tissue. More details about the PatchCamelyon dataset can be found here https://github.com/basveeling/pcam.

A subset of 10,000 images is provided. For 9000 of these you have labels which can be found in the train_labels_sample.csv file. The other 1000 images are a test or holdout set for which you do not have the labels. The test set predictions should be submitted alongside the notebook as a submission.csv file, and will be assessed using the labels we've removed. You will still need to split the 9000 images for which we provided labels into training and validation sets, for example using cross-validation.

Methods
Your project will have two parts:

Feature extraction. You can use any of the image feature extractors described in Lecture 8b such as SIFT, HoG, Daisy etc.
Classification. You can use any of the approaches in Lecture 7b such as Random Forest, Adaboost etc. You can try ensembling and/or stacking multiple classifiers, using multiple feature extractors, doing a search over hyperparameters etc.
Get started
Please download the data (.tif files) from https://drive.google.com/open?id=1jueMzoEGfjFI1nyJpmt9IpOwulH5oKlb

and place the files in a folder ./data. Then extract train_sample.zip to train_sample. You can then use the load_data_and_extract_features function to load the data and extract some features on the training data and test data.

Issues
For any issues with access to the data please contact thomas.varsavsky@kcl.ac.uk
