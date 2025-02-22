import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import pickle
"""#Load the Dataset"""

data = pd.read_csv('first inten project.csv')

# Data Cleaning and Preprocessing
data.columns = data.columns.str.strip()
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = data[col].replace(r'\s+', ' ', regex=True)

#Handle Outliers using IQR Method
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.clip(df[column], lower_bound, upper_bound)  # Cap values

outlier_columns = ["number of weekend nights", "number of week nights", "lead time", "average price"]
for col in outlier_columns:
    cap_outliers(data, col)

# Convert Date Columns
data['date of reservation'] = pd.to_datetime(data['date of reservation'], errors='coerce')

# Feature Engineering
data['reservation_month'] = data['date of reservation'].dt.month
data['reservation_year'] = data['date of reservation'].dt.year
data['reservation_day'] = data['date of reservation'].dt.day

# Handle Missing Values
data.fillna(data.median(numeric_only=True), inplace=True)

# Encode Categorical Features
categorical_cols = ["type of meal", "room type", "market segment type", "booking status"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

data = data.drop(["Booking_ID", "P-C", "P-not-C", "car parking space", "date of reservation", "repeated"], axis=1)

# Feature Scaling
X = data.drop(columns=['booking status'])
y = data['booking status']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# PCA for Dimensionality Reduction
pca = PCA(n_components=len(X.columns))
X_pca = pca.fit_transform(X_scaled)
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
X_pca_reduced = X_pca[:, :n_components]
feature_names = list(X.columns)

# Handle Imbalanced Data using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_pca_reduced, y)

# Compute Class Weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_resampled), y=y_resampled)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

data.head()

"""# Machine Learning Models

## Train models
"""

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# KNN Classifier
knn_params = {
    'n_neighbors': range(1, 10),
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
knn_model = KNeighborsClassifier()
grid_knn = GridSearchCV(knn_model, knn_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train, y_train)
best_knn = grid_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_report = classification_report(y_test, y_pred_knn)


"""##Model Evaluation

"""

print("KNN Model:")
print(f"Accuracy: {knn_accuracy}")
print(knn_report)

"""## Save Model, Scaler, PCA, and Feature Names"""

pickle.dump(best_knn, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))    
pickle.dump(pca, open('pca.pkl', 'wb'))
pickle.dump(feature_names, open('feature_names.pkl', 'wb'))