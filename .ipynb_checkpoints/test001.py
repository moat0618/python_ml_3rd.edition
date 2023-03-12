from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std, y_combined,clf=svm)