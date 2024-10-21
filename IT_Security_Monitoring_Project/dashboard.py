import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load the processed data with risk scores and anomalies
data = pd.read_csv('processed_security_logs.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

st.title("Enhanced IT Security Monitoring Dashboard")

# Display the log data with anomaly labels
st.subheader("Log Data with Anomaly Detection")
st.dataframe(data)

# Severity Distribution Pie Chart
st.subheader("Severity Distribution")
severity_counts = data['Severity'].value_counts()
fig, ax = plt.subplots()
ax.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%')
st.pyplot(fig)

# Daily Trend of Risk Scores
st.subheader("Daily Trend of Risk Scores")
avg_risk_score = data.groupby(data['Timestamp'].dt.date)['RiskScore'].mean()
st.line_chart(avg_risk_score)

# Number of Events per Day
st.subheader("Number of Events per Day")
daily_events = data.groupby(data['Timestamp'].dt.date)['EventType'].count()
st.bar_chart(daily_events)

# Failed Login Attempts by User
st.subheader("Failed Login Attempts by User")
failed_logins = data[(data['EventType'] == 'Login') & (data['Message'].str.contains('Failed'))]
failed_login_count = failed_logins['User'].value_counts()
st.bar_chart(failed_login_count)

# Anomaly Detection Results
st.subheader("Anomalies Detected")
anomalies = data[data['AnomalyLabel'] == 'Anomaly']
st.write(f"Total Anomalies Detected: {len(anomalies)}")
st.dataframe(anomalies[['Timestamp', 'User', 'EventType', 'Severity', 'Message']])

# Forecasting future trends using Prophet
st.subheader("Future Trend Prediction with Forecasting Details")

# Prepare data for forecasting
daily_counts = data.groupby(data['Timestamp'].dt.date)['EventType'].count().reset_index()
daily_counts.columns = ['ds', 'y']  # Prophet expects these column names

# Train the Prophet model
model = Prophet()
model.fit(daily_counts)

# Make a future dataframe for 30 days ahead
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Customize the forecast plot with labels and legend
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plot the forecasted trend
ax2.plot(forecast['ds'], forecast['yhat'], label='Forecasted Trend', color='blue')
ax2.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                 color='skyblue', alpha=0.4, label='Confidence Interval')

# Plot actual data points for reference
ax2.scatter(daily_counts['ds'], daily_counts['y'], color='red', label='Actual Event Count')

# Add titles and labels
ax2.set_title("Forecasted Security Event Trend for the Next 30 Days", fontsize=14)
ax2.set_xlabel("Date", fontsize=12)
ax2.set_ylabel("Number of Events", fontsize=12)

# Add a legend for clarity
ax2.legend(loc='upper left')

# Rotate the date labels for better readability
plt.xticks(rotation=45)

# Show the plot in Streamlit
st.pyplot(fig2)

# Add Interpretation Indices below the chart
st.write("""
### Forecast Interpretation:
- **Forecasted Trend (Blue Line):** The predicted number of security events for each day.
- **Actual Event Count (Red Dots):** Historical daily event counts for reference.
- **Confidence Interval (Shaded Area):** The range within which the forecast is likely to fall, with 95% confidence.

Use this forecast to anticipate future security events and take proactive actions to address potential risks.
""")
