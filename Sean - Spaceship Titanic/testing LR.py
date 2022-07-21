import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             plot_confusion_matrix, roc_auc_score, roc_curve)

from testing_data import *
from training_data import *

LR_model = LogisticRegression(max_iter=1000)
LR_model.fit(whole_train_X, whole_train_y)
predictions = LR_model.predict(finalX)

submission_predictions = pd.DataFrame(predictions)
submission_predictions.columns=['Transported']
print(submission_predictions)
result = pd.concat([IDs, submission_predictions], axis=1)
csv = result.to_csv("LRsubmission.csv", index=False)
print(csv)
