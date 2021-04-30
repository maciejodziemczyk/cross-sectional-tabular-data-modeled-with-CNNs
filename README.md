# Cross-sectional tabular data modeled with CNNs
Project created for Master Thesis at WNE UW.

Language:
 - English - notebooks
 - Polish - thesis

## About
In my Master Thesis I decided to use what I learned during my studies at WNE UW + something crazy. This crazy thing is modeling cross-sectional tabular data with convolutional neural networks + some other algorithms for comparison (logisitc regression, random forest, XGBoost and Multilayer Percepton). Data used in this project is about polish companies bankruptcy prediction task with financial indicators as predictors. The dataset contains 5 different forecast horizons, I wanted to analyze all of them (currently 3/5 done). 

I decided to perform train/test split, analyze the training one while test set is the "hold out" one. Firstly I perform some decoding stuff, balance and missings check. Then I usually had to delete some columns due to too many missing values (alwways the same attributes regardless of the forecast horizon). I imputed the rest via Random Forest data imputation algorithm developed during [Machine Learning 2 project](https://github.com/maciejodziemczyk/Can-PCA-extract-important-informations-from-non-significant-features-Neurak-Network-case).
