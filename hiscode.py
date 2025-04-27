import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Read the CSV file
df = pd.read_csv('input.csv')

# Drop the columns you want to ignore
df = df.drop(columns=['id', 'name', 'orbiting_body', 'sentry_object'])

# Separate features and target variable
X = df.drop(columns=['hazardous'])
y = df['hazardous']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Handle class imbalance on the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the Decision Tree Classifier with pruning
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# Train the classifier
clf.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Decision Tree Accuracy: {accuracy:.2f}')
print(f'Decision Tree Precision: {precision:.2f}')
print(f'Decision Tree Recall: {recall:.2f}')
print(f'Decision Tree F1 Score: {f1:.2f}')

# Try a Random Forest for comparison
rf_clf = RandomForestClassifier(random_state=42, max_depth=5)
rf_clf.fit(X_train_resampled, y_train_resampled)
rf_pred = rf_clf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f'Random Forest Accuracy: {rf_accuracy:.2f}')
print(f'Random Forest Precision: {rf_precision:.2f}')
print(f'Random Forest Recall: {rf_recall:.2f}')
print(f'Random Forest F1 Score: {rf_f1:.2f}')

# Visualize the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Not Hazardous', 'Hazardous'])