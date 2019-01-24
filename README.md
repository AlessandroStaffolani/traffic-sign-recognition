# Traffic Sign Recognition
This project is developed for the exam of Computer Vision at Alma Mater Studiorum of Bologna (UNIBO).

The aim of this project is to create an image classificator capable of reconize traffic signs. 

The dataset come from the famous german traffic sign benchmark [[link](http://benchmark.ini.rub.de/)], that provides more than 50 000 images relative to 43 different classes.

The project is composed of two targets:

1) Classify the traffic sign images that comes from the dataset, these images contains only the ROI (Region of Interest) and some pixel as contour. To achive this target I will use a Convolution Neural Network (CNN).
2) Find, from a real scene, where are traffic signs and than take them for classification using the previous point. To achive this target I will use Yolo algorithm.