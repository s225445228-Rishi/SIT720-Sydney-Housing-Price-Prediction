# SIT720 Sydney Housing Price Prediction

## Project Overview

This project was completed for the SIT720 Machine Learning mini project.

The aim of the project is to explore Sydney housing data and develop a machine learning model that can estimate property sale prices. The dataset contains 120 sold properties from Blacktown, Balmain and Bellevue Hill.

Three regression models were compared:
- Linear Regression
- Random Forest
- Gradient Boosting

Gradient Boosting was selected for the final prediction system.

## Project Files

- `SIT720_8.1D_Recent_Sydney_Housing_120_Properties.csv` - housing dataset used in the project
- `SIT720_8.1D_Rishi_Soni.ipynb` - data analysis and machine learning notebook
- `app.py` - Streamlit property price prediction application
- `requirements.txt` - Python packages required to run the application

## Running the Notebook

Download or clone the repository and open:

`SIT720_8.1D_Rishi_Soni.ipynb`

Run the notebook cells in order.

## Running the Web Application

Install the required packages:

`python3 -m pip install -r requirements.txt`

Then run:

`python3 -m streamlit run app.py`

The application will open in a web browser and allow property information to be entered to generate an estimated sale price.

## Note

The predictions in this project are for educational and decision-support purposes only and should not be considered professional property valuations.
