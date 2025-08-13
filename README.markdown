# Boston Housing Prices Prediction App

This project is a machine learning application that predicts house prices in Boston using a Random Forest regression model. The application is built with Streamlit and deployed on Streamlit Cloud.

## Project Structure 

```
boston-housing/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies for deployment
├── model.pkl            # Trained Random Forest model
├── project_report.pdf    # Project report
├── data/
│   └── boston.csv       # Boston Housing dataset
├── notebooks/
│   └── model_training.ipynb  # Jupyter notebook for EDA and model training
└── README.md            # Project documentation
```

   
## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd boston-housing

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Iris Dataset**:
   - The dataset is available from Kaggle: [Boston House Prices Dataset](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices).
   - Place `housing.csv` in the `data/` directory.

4. **Run the Application Locally**:
   ```bash
   streamlit run app.py
   ```

## Deployment Instructions
1. Push the repository to GitHub.
2. Connect the repository to Streamlit Cloud.
3. Ensure `requirements.txt` and `model.pkl` are included.
4. Configure the deployment settings in Streamlit Cloud.
5. Access the deployed app via the provided public URL.

## Features
- **Data Exploration**: View dataset shape, columns, data types, and filter by price range.
- **Visualizations**: Interactive histograms and scatter plots using Plotly to explore feature distributions and relationships with price.
- **Model Prediction**: Input values for 13 features to predict house prices in real-time.
- **Model Performance**: Displays RMSE, R² score, and a feature importance bar chart for the Random Forest model.

## Dependencies
Listed in `requirements.txt`:
- streamlit==1.24.0
- pandas==2.0.3
- plotly==5.15.0
- scikit-learn==1.3.0
- numpy==1.25.0

## Model Details
- **Dataset**: Boston Housing dataset (506 samples, 13 features, 1 target: median house price in $1000s).
- **Models Trained**: Random Forest Regressor and Linear Regression.
- **Best Model**: Random Forest (selected based on cross-validation R² score of 0.83).
- **Preprocessing**: No missing values; features used as-is due to their relevance.
- **Evaluation Metrics**:
  - Test RMSE: 3.17
  - Test R²: 0.85


## Deployment Notes
- The model file (`model.pkl`) is below GitHub's 10MB limit, so Git LFS is not required.
- Ensure the `data/boston.csv` file is available or provide download instructions in the app.

## Streamlit Cloud URL
https://savindhayachamikara-machine-learning-model-deployment.streamlit.app/

## Author
I.R.G.S.C.Herath
ITBIN-2211-0193
