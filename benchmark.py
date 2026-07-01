from sklearn.svm import SVC
from data_loader import X_train, y_train, X_test, y_test, get_batch
from model import init_params, forward, loss_function, backward, update_params
from train import train, evaluate

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
svm_accuracy = svm.score(X_test, y_test)
print(f"SVM accuracy: {svm_accuracy:.4f}")
print(f"MLP accuracy: {accuracy:.4f}")
