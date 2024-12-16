import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# טען את הנתונים
df = pd.read_csv('data/data.csv')

# הכן את הנתונים
X = df[['MonthlyIncome', 'WorkLifeBalance', 'TotalWorkExperience']]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# אמן מודל בוסטינג של גרדיאנט
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# הערך את הביצועים
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}, F1-score: {f1:.2f}')

# שמור את המודל המאומן
import joblib
joblib.dump(model, 'models/churn_model.pkl')