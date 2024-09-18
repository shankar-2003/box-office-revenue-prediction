import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os

# Load the dataset
df = pd.read_csv(r"C:\Users\Admin\anaconda3\Lib\site-packages\pandas\io\parsers\movies_metadata.csv", low_memory=False)

# Convert relevant columns to numeric, forcing errors to NaN
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

# Select relevant columns
df = df[['budget', 'revenue', 'runtime', 'popularity']]

# Drop rows with missing values
df = df.dropna()

# Define features (X) and target (y)
X = df[['budget', 'runtime', 'popularity']]
y = df['revenue']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Initialize FastAPI
app = FastAPI()

# Define a Pydantic model for input data validation
class MovieData(BaseModel):
    budget: float
    runtime: float
    popularity: float

# Create a prediction endpoint
@app.post("/predict")
def predict_revenue(data: MovieData):
    # Prepare the data for prediction
    input_data = np.array([[data.budget, data.runtime, data.popularity]])
    
    # Make prediction
    predicted_revenue = model.predict(input_data)[0]
    
    return {"predicted_revenue": predicted_revenue}

# Endpoint to generate and serve "Actual vs Predicted" plot
@app.get("/plot/actual_vs_predicted")
def plot_actual_vs_predicted():
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='purple', linewidth=2)
    plt.title('Actual vs Predicted Box Office Revenue')
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    
    # Save the plot as an image
    plot_path = "actual_vs_predicted.png"
    plt.savefig(plot_path)
    plt.close()
    
    return FileResponse(plot_path, media_type="image/png")

# Endpoint to generate and serve "Residuals Distribution" plot
@app.get("/plot/residuals")
def plot_residuals():
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    
    # Save the plot as an image
    plot_path = "residuals_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    
    return FileResponse(plot_path, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
