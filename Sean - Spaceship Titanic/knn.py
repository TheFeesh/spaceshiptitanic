from sklearn.neighbors import KNeighborsClassifier
from testing_data import *
from training_data import *
from sklearn import metrics
import matplotlib.pyplot as plt


# print(train_X.shape)
# print(test_X.shape)
# print(train_y.shape)
# print(test_y.shape)

# testing k values
# k_range = range(1,30)
# scores = {}
# scores_list = []
# for k in list(k_range):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(train_X, train_y)
#     y_pred=knn.predict(test_X)
#     scores[k] = metrics.accuracy_score(test_y, y_pred)
#     scores_list.append(metrics.accuracy_score(test_y, y_pred))

# plt.plot(k_range, scores_list)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Testing Accuracy')
# plt.show()

# actual fitting model
knn_model = KNeighborsClassifier(n_neighbors=21)
knn_model.fit(whole_train_X, whole_train_y)
predictions = knn_model.predict(finalX)

submission_predictions = pd.DataFrame(predictions)
submission_predictions.columns=['Transported']
print(submission_predictions)
result = pd.concat([IDs, submission_predictions], axis=1)
csv = result.to_csv("KNNsubmission.csv", index=False)