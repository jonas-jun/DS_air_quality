# DS_air_quality

2021-06  
mini project to predict air quality in Seoul

### Goal
Bad Recall score: true positive bad air quality / truth bad air quality(bad, worst)

### Models
- Random Forest: variate max_depth and class_weight
- LSTM, att: bidirectional, the number of sequences to input

### Contents  
- RF_predictor.ipynb - main of Random Forest model (non-seq)  
- AiR_predictor.ipynb - main of LSTM, att model, run at colab but possible at any gpu env. (seq)  
- prepare_dataset.ipynb - merge, EDA, feature engineering  
- *.py should be used for *.ipynb files  

Random Forest with class_weight {1,1,3,5} is the best.


datasets in this drive  
[link](https://drive.google.com/drive/folders/1yzA9S7DetYa86z7nF59CTyamYjomVySE?usp=sharing)
