import pandas as pd

# Load the data
df = pd.read_csv(r"C:\Users\hp\Downloads\NqndMEyZakuimmFI.csv")

# 1. Drop irrelevant columns
columns_to_drop = ['job_id', 'salary_range', 'telecommuting', 'has_company_logo', 
                   'has_questions', 'employment_type', 'required_experience', 
                   'required_education', 'industry', 'function', 'company_profile']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

# 2. Drop rows with missing values in key columns
df.dropna(subset=['title', 'location', 'description'], inplace=True)

# 3. Combine text columns into one for easier processing
df['text'] = df['title'].astype(str) + ' ' + df['location'].astype(str) + ' ' + df['description'].astype(str)

# 4. Encode target variable (fraudulent column)
df['fraudulent'] = df['fraudulent'].astype(int)

# 5. Show result
print("✅ Data cleaned successfully!")
print(df[['text', 'fraudulent']].head())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# 1. Vectorize the text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['fraudulent']

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Evaluate model
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)

print(f"\n✅ Model trained! F1-Score: {f1:.4f}")

import joblib  # Make sure it's imported at the top if not already

# Save the model and vectorizer
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")


# Load test data
test_df = pd.read_csv(r"C:\Users\hp\Desktop\tho\test.csv.csv")  # Update path if different

# Fill missing text fields (same as before)
test_df.fillna("", inplace=True)

# Combine text features just like training
test_df["combined"] = test_df["title"] + " " + test_df["description"] + " " + test_df["company_profile"]

# Use the same vectorizer (DO NOT FIT AGAIN)
X_test = vectorizer.transform(test_df["combined"])

# Predict fraud probability and labels
probs = model.predict_proba(X_test)[:, 1]  # Get probability of being fraud
preds = model.predict(X_test)              # Get final prediction (0 or 1)

# Add predictions to the test data
test_df["fraudulent"] = preds
test_df["fraud_probability"] = probs

# Print sample results
print(test_df[["title", "fraudulent", "fraud_probability"]].head(10))



