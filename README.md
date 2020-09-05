# Project definition
Weather Prediction Using Deep Learning Techniques


# Presented by
15BCE120 – Meera Suthar
15BCE126 – Yash Thesia
15BCE130 – Vidhey Oza


# Abstract

This term paper surveys various research papers aimed at forecasting weather. We explore cutting-edge research towards deep learning algorithms and technology. Convolutional networks help preserve information of data points in physical or logical proximity when placed together on a matrix, and help reduce the computation size using the concept of 2-D filters, for example precipitation levels of a location based on levels of neighboring cities or towns. Recurrent networks help preserve information gained with past iterations of training, and are thus much more flexible when trying to analyze time-series data, like today’s precipitation levels based on past precipitation levels. And autoencoders help overcome overfitting by extracting useful features from the given data, which also helps reduce the vanishing gradient problem.


# Libraries required
numpy, pandas, matplotlib, tensorflow, keras, scikit-learn, their dependencies, and other built-in libraries.

# Files/directories in package 

WeatherDL – contains python scripts.
1. model_maker.py – contains different implementations as discussed in term paper. 
2. data_maker.py – data formatting python scripts (in format required by model_maker.py). 
3. sample.py – RUN THIS FILE. 

dataset – contains date-wise CSV data for different locations. Obtained from Dr. Sanjay Garg as part of Minor Project with Vidhey Oza & Dhananjay Rasalia

weather.gif – animation of heatmaps of meteorological variables over a 50-day time period. 


*Please refer to inline comments in python scripts for detailed descriptions.*

*Please change paths in data_maker.py, as complete paths may have to be given for the code to run properly.*

