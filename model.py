from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Load your dataset
data = pd.read_excel('encoded_data.xlsx')


selected_feature =data[[
    'Loan Amount Requested (INR)','Annual Income (INR)','Credit Score','Farm Size (acres)',
    'Type of Crop(s)_Groundnut','Type of Crop(s)_Maize', 'Type of Crop(s)_Rice',
       'Type of Crop(s)_Soybean', 'Type of Crop(s)_Sugarcane',
       'Type of Crop(s)_Wheat',
    'Yield per Acre (tons/acre)',
    'Market Prices (INR/ton)',
    'Inflation Rate (%)',
    'Age','Loan Eligibility',
    'Loan Purpose_Debt Consolidation',
       'Loan Purpose_Farm Equipment', 'Loan Purpose_Irrigation',
       'Loan Purpose_Seeds & Fertilizers',
    'Current Debt Level (INR)','Previous Loan History'
]]
target_column = 'Loan Eligibility'


X=selected_feature.drop(['Loan Eligibility'],axis=1)
y=selected_feature['Loan Eligibility']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)



predict_model=rf_model.predict(X_test)
accuracy = accuracy_score(y_test, predict_model)
print(f"Accuracy: {accuracy}")

# Save the trained model
joblib.dump(rf_model, 'random_forest_model.pkl')

print("✅ Model trained and saved successfully!")