Once you have downloaded the PASCAL face dataset from the Caltech website 
(as for May 3rd 2022, Caltech is doing data migration therefore the data is unavailable to the public) 

please run the DataTreatment.ipynb within the dataset folder to generate training data. 

Make sure the ImageData.mat is in the same directory as DataTreatment.ipynb

To train the yolo model, please run train.ipynb file

To test the model, please run sample.ipynb file

Utility.py, loss.py, and dataset.py are all from the machine learning collection repository by Aladdin Persson
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO



* We did not upload models and datasets