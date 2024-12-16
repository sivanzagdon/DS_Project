import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# טען את הנתונים
df = pd.read_csv('data/data.csv')

# הכן את הנתונים
X = df[['JobLevel', 'EducationField', 'MonthlyIncome']]
y = df['PerformanceScore']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# אמן מודל רער יערות אקראי
model = RandomForestRegressor()
model.fit(X_train, y_train)

# הערך את הביצועים
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse:.2f}, R-squared: {r2:.2f}')

# שמור את המודל המאומן
import joblib
joblib.dump(model, 'models/performance_model.pkl')