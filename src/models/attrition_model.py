import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import joblib

# קריאת הקובץ
df = pd.read_csv('C:/Users/Sivan Zagdon/DS/Project/DS_Project/src/data/data.csv')

# המרת הערכים 'No' ו-'Yes' ל-0 ו-1
df['Attrition'] = df['Attrition'].map({'No': 0, 'Yes': 1})

# הגדרת משתנים
X = df[['YearsSinceLastPromotion', 'Age', 'YearsInCurrentRole', 'YearsAtCompany', 'TotalWorkingYears']]
y = df['Attrition']

# חלוקה לסטים של אימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# הגדרות עבור הרשת הנוירונית
input_layer = X_train.shape[1]  # מספר התכונות
hidden_layer = 128  # מספר הנוירונים בשכבה הסמויה
output_layer = 1  # מספר הנוירונים בשכבת הפלט (תוצאה בינארית)

# אתחול המטריצות
W1 = np.random.randn(hidden_layer, input_layer)  # משקולות עבור השכבה הראשונה
b1 = np.zeros((hidden_layer, 1))  # ביוס עבור השכבה הראשונה
W2 = np.random.randn(output_layer, hidden_layer)  # משקולות עבור השכבה השנייה
b2 = np.zeros((output_layer, 1))  # ביוס עבור השכבה השנייה

# פונקציית סיגמואיד
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # הגבלת הערכים כדי למנוע overflow

# פונקציה לעדכון שיעור הלמידה
def adjust_learning_rate(initial_lr, epoch, decay_rate=0.1, decay_steps=100):
    """
    Adjust the learning rate dynamically as the training progresses.
    
    Parameters:
    - initial_lr: The initial learning rate.
    - epoch: The current epoch number.
    - decay_rate: The factor by which to decay the learning rate. Default is 0.1.
    - decay_steps: The number of epochs after which to apply decay. Default is 100.
    
    Returns:
    - The updated learning rate.
    """
    if epoch % decay_steps == 0 and epoch > 0:
        new_lr = initial_lr * (1 / (1 + decay_rate * (epoch // decay_steps)))
        return new_lr
    return initial_lr

# הגדרת מאפיינים
epochs = 1000
learning_rate = 0.01
num_of_examples = X_train.shape[0]  # מספר הדוגמאות מתוך X_train

# המרת X_train ו-Y_train למערכים של NumPy
X_train_np = X_train.values.T  # המרת X לארגון שנוכל לעבוד איתו
Y_train_np = y_train.values.reshape(1, -1)  # המרת Y גם לערך דו-ממדי

# לימוד הרשת
for epoch in range(epochs):
    # עדכון שיעור הלמידה כל 100 אפוכות
    learning_rate = adjust_learning_rate(learning_rate, epoch)

    # הפצה קדימה
    Z1 = np.dot(W1, X_train_np) + b1  # חישוב משקולות + bias
    A1 = sigmoid(Z1)  # הפעלת פונקציית סיגמואיד
    Z2 = np.dot(W2, A1) + b2  # חישוב פלט
    A2 = sigmoid(Z2)  # הפעלת פונקציית סיגמואיד

    # חישוב הפסד (log loss)
    loss = -np.mean(Y_train_np * np.log(A2) + (1 - Y_train_np) * np.log(1 - A2))

    # עדכון משקולות באמצעות Backpropagation
    dZ2 = A2 - Y_train_np
    dW2 = np.dot(dZ2, A1.T) / num_of_examples
    db2 = np.sum(dZ2, axis=1, keepdims=True) / num_of_examples
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = np.dot(dZ1, X_train_np.T) / num_of_examples
    db1 = np.sum(dZ1, axis=1, keepdims=True) / num_of_examples

    # עדכון משקולות
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# הפצה קדימה על קבוצת הטסט
X_test_np = X_test.values.T
Z1_test = np.dot(W1, X_test_np) + b1
A1_test = sigmoid(Z1_test)  # הפעלת פונקציית סיגמואיד לשכבה הראשונה
Z2_test = np.dot(W2, A1_test) + b2
A2_test = sigmoid(Z2_test)  # הפעלת פונקציית סיגמואיד לשכבה השנייה (פלט סופי)

# תחזיות על בסיס הפלט
predictions = (A2_test > 0.5).astype(int)  # הפוך את תחזיות הפלט לעמדה בינארית
labels = (y_test.values.reshape(1, -1) > 0.5).astype(int)  # הפוך את התוצאות האמיתיות לעמדה בינארית

# הדפסת מטריצת בלבול
print("Confusion Matrix:")
print(confusion_matrix(predictions.T, labels.T))

# חישוב דיוק
accuracy = np.mean(predictions == labels) * 100
print(f"Accuracy: {accuracy:.2f}%")

# שמירת המודל
model_params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
joblib.dump(model_params, './attrition_model_nn_custom.pkl')  # שמירת המודל
