from training_data import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix


LR_model = LogisticRegression(max_iter=1000)
LR_model.fit(train_X, train_y)
predictions = LR_model.predict(test_X)

# print(LR_model.score(validation_X, validation_y))

probs = LR_model.predict_proba(test_X)
probs = probs[:, 1]
fpr, tpr, thresholds = roc_curve(test_y, probs)

cm = confusion_matrix(test_y, predictions)
# print(cm)

# print(classification_report(test_y, predictions))