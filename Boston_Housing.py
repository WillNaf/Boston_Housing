import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('/Users/williamnufziger/desktop/joe/Boston.csv') 

# Histogram or Density Plot of Target Variable (`MEDV`)
plt.hist(df['MEDV'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of MEDV')
plt.xlabel('MEDV')
plt.ylabel('Frequency')
plt.savefig('images/histogram.png')
plt.show()

X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT']]
X = sm.add_constant(X)
y = df['MEDV']

model = sm.OLS(y, X).fit()
y_pred = model.predict(X)

plt.scatter(y_pred, y - y_pred, alpha=0.5)
plt.title('Residual Plot')
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.savefig('images/scatter.png')
plt.show()

rmse = mean_squared_error(y, y_pred, squared=False)
print("RMSE: ", rmse)

print(model.summary())

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True)
plt.savefig('images/correlation_matrix.png')
plt.show()

for col in X.columns:
    if col != "const":
        sns.boxplot(x=df[col], y=df['MEDV'])
        plt.title(f'MEDV vs {col}')
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        plt.savefig(f'images/box_{col}.png')
        plt.show()

X = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT']]
y = df['CAT.MEDV']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

tree.plot_tree(clf)
plt.savefig('images/Tree.png')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=400, test_size=106, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Feature Importance for Decision Tree
importances = clf.feature_importances_
features = X.columns
plt.barh(features, importances, align='center')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.savefig('images/tree_importance.png')
plt.show()

# Decision Tree Depth vs. Accuracy Plot
depths = list(range(1, 11))  # Adjust range accordingly
accuracies = []
for depth in depths:
    clf_temp = tree.DecisionTreeClassifier(max_depth=depth)
    clf_temp = clf_temp.fit(X_train, y_train)
    accuracies.append(clf_temp.score(X_test, y_test))
plt.plot(depths, accuracies, '-o')
plt.title('Tree Depth vs. Accuracy')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.savefig('images/tree_accuracy.png')
plt.show()

# Confusion Matrix Visualization
confusion = confusion_matrix(y_test, y_pred)
sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('images/confusion.png')
plt.show()
