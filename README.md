# Cross-sectional tabular data modeled with CNNs
Project created for Master Thesis at WNE UW.

Language:
 - English - notebooks
 - Polish - thesis

## About
In my Master Thesis I decided to use what I learned during my studies at WNE UW + something crazy. This crazy thing is modeling cross-sectional tabular data with convolutional neural networks + some other algorithms for comparison (logisitc regression, random forest, XGBoost and Multilayer Percepton). Data used in this project is about polish companies bankruptcy prediction task with financial indicators as predictors. The dataset contains 5 different forecast horizons, I wanted to analyze all of them (currently 3/5 done). 

I decided to perform train/test split, analyzed the training one while test set was the "hold out" one. Firstly I performed some decoding stuff, balance and missings check. Then I usually had to delete some columns due to too many missing values (always the same attributes regardless of the forecast horizon). I imputed the rest via Random Forest data imputation algorithm developed during [Machine Learning 2 project](https://github.com/maciejodziemczyk/Can-PCA-extract-important-informations-from-non-significant-features-Neurak-Network-case). After that I plotted some distribution plots (box plots and histograms) and realized the big outliers problem. To solve outliers problem 
I decided to clip the data with quantiles (0.005 and 0.995), I found the idea in Pawełek (2019). After that I plotted distributions again and foud big skewness, to reduce it I found Yeo-Johnson transformation very useful (idea found in Son et. al 2019). After power trasform I normalized the data. Next, I decided to perform 6-fold stratified Cross Validation study on the training set and the whole preprocessing pipeline was applied for all CV folds separately (imputation, quantile clipping, power transform and normalization). 
I found the whole preprocessing pipeline very useful for logit MLP and CNN. I treated it as hyperparameter for tree-based methods. RF and XGBoost hyperparameters was optimized as always - firstly sensivity to hparam change was inspected to define the ranges, next random search was performed. For Logit I tried different L1 regularization coefs. For neural networks I performed experimets and looked on learning curves and results. <br>
The very interesting part of my study was to convert observation vector to the image to be a proper input for convolutional neural network. The whole how to, was based on the correlation matrix and monte carlo optimization -> I reshaped vector to the matrix where one variable was one matrix element, after that I performed quasi optimization process via Monte Carlo simulation to make more correlated features closer to each other (idea found in Hosaka 2019). The story behind is "when we see black cat sitting on the white pillow then P(x_i=0|x_j=0) > P(x_i=0|x_j=1)". My algorithm for optimization had to be fast (I had big plans for the future); my last version based on precomputed correlation matrix and python dictionaries was approx 150 times faster than the first one based on numpy.arrays. Matrices had to be normalized to 0-255 range. Because I had 59 features in total my images was 8x8 pixels I decided to enlarge it to the 20x20 (this is the big plan mentioned before). To do so I built a discrete probability distribution where features was events and its XGBoost feature importance score folds mean was the event probability (I had to normalize it to sum up to 1). Then, I sampled two features from created distribution and one basic arithmetic operation (add, subtract, multiply, divide) from uniform distribution (Idea found in Zięba et. al. 2016).
I had to prevent dividing by 0, and samplig the same two features with the same operation. My new feature was of course the composition. I repeated sampling operation 341 timmes to get 400 features. I saw another effect of my preprocessing pipeline - more components more sharp and clear images. <br>
After Cross validation study I compared the results and performed wilcoxon test where I found it useful (to choose between 8x8 and 20x20 CNN). When I specified the best logit, RF, XGBoost, MLP and CNN I trained them on the whole training set (preprocesing had to be performed again of course). Models were evaluated on the test set.

main Findings:
 - outliers reduction, power trasform and normalization are crucial steps while dealing with financial indicators especially for derivatives based models (logit, nets)
 - MLP is an awesome model I noted more than 0.7 AUC-PR on the test set on 1 year horizon and more than 0.6 for 3 years horizon (currently I'm working on 4- and 5- year horizons)
 - proposed procedure to model with CNN is quite good, worse than MLP, but it was 2nd or 3rd best algorithm with pretty good results (approx 0.58 AUC-PR on 1 year horizon) 
 - inception modules for CNNs were very useful
 - longer forecast horizon, not always worse results :)

In this project I learnt a lot, a lot about python coding, a lot about neural nets, convolutions (theory and practice), a lot about keras (functional tf API), a lot about data anslysis and more. This is my biggest project so far I guess.

## Repository Description
 - Magisterka_Xyear_all.ipynb - Notebook for X forecast horizon
 - data folder - all the data sets - orginal, dumped preprocessed, splitted folds (I had to perform it once and load during validation to made this research possible)
 - results folder - all the saved models

## Technologies
 - Python (pandas, numpy, scipy, matplotlib.pyplot, PIL, statsmodels, scikit-learn, xgboost, keras)
 - Jupyter Notebook
 - LaTeX (Thesis)

## Author
 - Maciej Odziemczyk

## Note
I can't describe whole analysis in just readme, I encourage you to check it by yourself, particular notebooks are pretty simillar to each other. 
