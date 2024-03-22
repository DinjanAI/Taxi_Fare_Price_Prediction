# Taxi_Fare_Price_Prediction

This project aims to predict taxi fare prices in New York City using machine learning techniques. By leveraging historical taxi trip data from NYC, we aim to build a model that accurately estimates the fare for a given trip based on various factors such as distance, time, location, and other relevant features.

## Dataset

The dataset used for this project consists of historical taxi trip records in New York City. It includes features such as pickup and dropoff coordinates, pickup datetime, trip distance, and fare amount. The dataset is preprocessed and cleaned to remove any inconsistencies or outliers that may affect the model's performance.

## Setting Up Your Environment

To run this Taxi Fare Price Prediction, you'll need a virtual environment to manage dependencies effectively. Here's a step-by-step guide:

**Prerequisites:**

- Python 3.7+ (Check version: `python --version` in your terminal.)
- `venv` module included with Python 3.3+ ([Link to instructions](https://docs.python.org/3/library/venv.html))

**Steps:**

1. **Open a terminal/command prompt.**
2. **Navigate to your project directory:** Use `cd <path_to_project_directory>`.
3. **Create a virtual environment:**
   - PowerShell: `python -m venv myenv`
   - CMD: `python -m venv myenv` (Replace `myenv` with your desired name.)
4. **Activate the virtual environment:**
   - PowerShell: `.\myenv\Scripts\activate`
   - CMD: `myenv\Scripts\activate`
5. **Install dependencies:** `pip install -r requirements.txt` (Download `requirements.txt` if not present.)

**Troubleshooting:**

- If `venv` is missing, install it using `pip install virtualenv` within the activated virtual environment.
- For issues with Python version or `pip` installation, refer to the documentation.

**Running the Chatbot:**

1. **Within the activated virtual environment:**
2. **Run the command:** `python nyc.py`


