from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

digits=load_digits()

X=digits.data
y=digits.target

X_train,X_cv,y_train,y_cv=train_test_split(X,y,test_size=0.20, random_state=1)

model=KNeighborsClassifier();

model.fit(X_train,y_train)
predictions=model.predict(X_cv)

print("\nAccuracy score: ", accuracy_score(y_cv, predictions))
print("\nConfusion matrix: \n", confusion_matrix(y_cv, predictions))
print("\nClassification report: \n", classification_report(y_cv, predictions))
