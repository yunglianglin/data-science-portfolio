##  💡 Detecting Loan Intent from Behavioral Sequences in Banking App

## 📝 Overview
Many customers lack sufficient transactional data, making traditional cross-selling models ineffective. This project introduces an LSTM-based sequential model that learns loan intent from digital behavioral patterns in clickstream data.

The pipeline embeds page names and integrates sequential features such as dwell time, action count, and loan-page visits, with padded sessions to handle variable sequence lengths. These inputs are processed through bidirectional and stacked LSTM layers with an attention mechanism to highlight critical steps, while a multi-layer perceptron (MLP) combines static features to capture non-linear relationships.

The model is trained efficiently using batch processing, sequence masking, and GPU-optimized techniques such as autocast and GradScaler, enabling scalable and fast experimentation.
##
![Overview 1](overview1.png)
##  
![Overview 2](overview2.png)


## 📂 Files
**Python_model_training**</br> 
This code snippet covers the process of building a LSTM model:
1. Sample Preprocessing: Prepare sequential and static features, apply padding, and record the actual sequence lengths.
2. Data Splitting: Partition the dataset into training, validation, and test sets to ensure reliable model evaluation.
3. DataLoader Construction: Implement custom collate functions to batch variable-length sequences while preserving their true lengths.
4. Model Training: Train the LSTM model on GPU, monitoring validation metrics to fine-tune performance and prevent overfitting.
