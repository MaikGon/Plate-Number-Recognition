# Plate-Number-Recognition

Project was made for computer vision classes.  
The template comes from [here](https://github.com/PUTvision/ImageProcessingCourse/tree/master/project_template_2020).

The goal was to find a car plate on an image and recognize signs on the extracted plate. The main code is in the 'utils.py' file. 
To recognize your plates, you should run 'main.py' with two arguments:
+ folder with car images
+ json file to save results to

Project assumptions:
+ 7 characters on the plate
+ width of the plate should not be greater than 1/3 of the image's width
+ rotation of the plate should not be greater than 45 degrees
+ maximum processing time per image is 2 seconds

Requirements:
+ python 3.7
+ scikit-image 0.17.2
+ opencv 4.2
