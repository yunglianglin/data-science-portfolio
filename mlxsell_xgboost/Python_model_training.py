#########----- 1. Retrieving data from a database ---------- #######
import pyodbc
import pandas as pd
import numpy as np

# connect to database
con = pyodbc.connect('Driver={SQL Server}; Server=###.##.###.##; Database=db_nm; Trusted_Connection=True')
cursor = con.cursor()

# query feature table
query = '''
        SELECT * 
        FROM tb_ml_wm_var
        '''
df_features = pd.read_sql(query, con)



#########----- 2. Splitting the data into training, validation, and test sets ---------- #######
# split into features and target
X = df_features.drop(columns=[
    'case_no',
    'cust_id',
    'yn_event'
])
y = df_features['yn_event']


# stratified split: train 70%, test 30%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.3
)

# reset index
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)



#########----- 3. Training segment-specific XGBoost models with Optuna for hyperparameter tuning  ---------- #######
from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import numpy as np

# Because the data is imbalanced, we use scale_pos_weight
# Count the number of samples in each class
class_counts = np.bincount(y_train)
# Get the number of samples in the majority class
majority_count = np.max(class_counts)
# Get the number of samples in the minority class
minor_count = np.min(class_counts)
# Calculate scale_pos_weight
scale_pos_weight = majority_count / minor_count


# Define Optuna's objective function
def objective(trial):
    # Suggest hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.3),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'scale_pos_weight': trial.suggest_uniform(1 * scale_pos_weight, 20 * scale_pos_weight),  # Range around calculated weight
        'tree_method': 'hist',
        'device': 'cuda'
    }
    
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    
    cv = StratifiedKFold(n_splits=5)
    f1_scores = []
    
    for train_idx, valid_idx in cv.split(X_train, y_train):
        X_fold_train, X_fold_valid = X.iloc[train_idx], X.loc[valid_idx]
        y_fold_train, y_fold_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_valid)
        
        # Calculate metrics
        f1 = f1_score(y_fold_valid, y_pred)
        prec = precision_score(y_fold_valid, y_pred)
        auc = roc_auc_score(y_fold_valid, y_pred)
        print('Precision: %5f, F1: %5f, AUC: %5f' % (prec, f1, auc))
        f1_scores.append(f1)
    
    return np.mean(f1_scores) #use f1 as the objective of Optuna to be maximized


# Create Optuna study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print("Best hyperparameters:", study.best_params)

# Retrieve the best parameters found by Optuna and initialize a new XGBoost model with them
best_params = study.best_params
best_model = XGBClassifier(**best_tree_params, use_label_encoder=False, eval_metric='logloss')
best_model.fit( X_train, y_train,  eval_set = [(X_train, y_train),(X_test, y_test)])



#########----- 4. Explaining model predictions using SHAP  ---------- #######
import shap
# Initialize SHAP explainer for the trained model
explainer_model = shap.Explainer(best_model)

# Compute SHAP values for the training set
shap_values = explainer_model(X_train)

# Display mean absolute SHAP values in a bar plot (feature importance)
shap.summary_plot(shap_values, X_train, plot_type="bar")

# Create a DataFrame of average absolute SHAP values per feature
mean_shap_values = pd.DataFrame({
    'feature': X_train.columns,
    'shap_value_mean': np.abs(shap_values.values).mean(axis=0)
})



#########----- 5. Evaluating model performance with metrics like F1, ROC-AUC, and PR-AUC  ---------- #######
# Evaluate model performance using multiple classification metrics
def evaluate_metrics(X_val, y_val, model):
    y_pred = model.predict(X_val)
    
    # Precision: Among all predicted positives, how many are actually positive
    precision = precision_score(y_val, y_pred)
    
    # Recall: Among all actual positives, how many are correctly identified
    recall = recall_score(y_val, y_pred)
    
    # F1 Score: Harmonic mean of precision and recall (balance between both)
    f1 = f1_score(y_val, y_pred)
    
    # Accuracy: Proportion of correctly predicted samples (both 0 and 1)
    accuracy = accuracy_score(y_val, y_pred)
    
    # ROC AUC: Area under the ROC curve using predicted probabilities
    y_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_proba)
    
    print(
        "precision:", precision,
        "recall:", recall,
        "f1:", f1,
        "accuracy:", accuracy,
        "auc:", auc
    )
    
    return y_pred, y_proba



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
import matplotlib.pyplot as plt

# Evaluate model performance on the training set
evaluate_metrics(X_train, y_train, best_tree_model)

# Evaluate on the training set to check model validity
ConfusionMatrixDisplay.from_estimator(
    best_tree_model,
    X_train, #X_test
    y_train, #y_test
    values_format='d',
    display_labels=['M', 'A'],  
    cmap=plt.cm.Blues
)



#########----- 6. Creating binners for production use  ---------- #######
## Evaluate Binning Performance
from sources.binner import NumericFeatureBinner

# Predict probability for the positive class (1)
train_proba = pd.DataFrame(best_model.predict_proba(X_train)[:, 1], columns=['proba_1']).reset_index(drop=True)
test_proba = pd.DataFrame(best_model.predict_proba(X_test)[:, 1], columns=['proba_1']).reset_index(drop=True)

# Prepare target variable for training and test sets
df_y_train = pd.DataFrame(y_train, columns=['target']).reset_index(drop=True).astype(float)
df_y_test = pd.DataFrame(y_test, columns=['target']).reset_index(drop=True).astype(float)

# Initialize numeric binner and fit using percentile-based binning (max 10 bins)
binner_model = NumericFeatureBinner()
binner_model.fit(train_proba, df_y_train, max_bins=10, method='percentile')

# Plot binning performance on training and test sets
binner_model.plot({'training_set': [train_proba, df_y_train]})
binner_model.plot({'test_set': [test_proba, df_y_test]})



#########----- 7. Saving and loading trained models for deployment  ---------- #######
# Save the trained model to a file
with open('D:/***/****/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Load the saved model from the file
with open('D:/***/****/best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

