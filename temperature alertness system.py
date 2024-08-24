import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load historical weather data (replace with actual dataset)
# Assuming the dataset has columns like 'date', 'temperature', 'humidity', etc.
weather_data = pd.read_csv('historical_weather_data.csv')

# Basic data analysis
print(weather_data.head())
print(weather_data.describe())

# Visualize temperature trends
plt.figure(figsize=(10, 6))
sns.lineplot(x='date', y='temperature', data=weather_data)
plt.title('Temperature Trends Over Time')
plt.show()

# Identify potential heatwave days (e.g., days with temperature above a certain threshold)
heatwave_days = weather_data[weather_data['temperature'] > 35]  # Example threshold
print(f"Number of heatwave days: {len(heatwave_days)}")
//other predictive code

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Feature engineering
weather_data['is_heatwave'] = weather_data['temperature'] > 35

# Prepare features and target variable
X = weather_data[['temperature', 'humidity', 'wind_speed', 'pressure']]
y = weather_data['is_heatwave']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
//mobile alert integration
from twilio.rest import Client

# Twilio configuration (replace with your actual credentials)
account_sid = 'your_account_sid'
auth_token = 'your_auth_token'
client = Client(account_sid, auth_token)

def send_heatwave_alert(phone_number, message):
    client.messages.create(
        body=message,
        from_='+1234567890',  # Replace with your Twilio number
        to=phone_number
    )

# Example: Sending an alert if a heatwave is predicted for tomorrow
tomorrow_weather = X_test.iloc[0]  # Example: using the first row of test set
if model.predict([tomorrow_weather])[0]:
    send_heatwave_alert('+19876543210', 'Alert: Heatwave expected tomorrow. Stay hydrated and indoors!')

print("Alert sent.")
