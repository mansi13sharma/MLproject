# Student Performance Prediction

A machine learning project that predicts student math scores based on various demographic and educational factors. The project implements multiple regression models with hyperparameter tuning and provides a Flask-based web interface for predictions.

## Features

- Predicts student math performance based on:
  - Gender
  - Race/Ethnicity
  - Parental level of education
  - Lunch type (standard/free-reduced)
  - Test preparation course completion
  - Reading score
  - Writing score

- Multiple ML models with hyperparameter tuning:
  - Random Forest Regressor
  - Decision Tree Regressor
  - Gradient Boosting Regressor
  - Linear Regression
  - XGBoost Regressor
  - CatBoost Regressor
  - AdaBoost Regressor

- Web-based user interface for easy predictions
- Modular codebase with separate components for data ingestion, transformation, and model training
- Automated model selection based on R² score

## Project Structure

```
MLproject/
│
├── app.py                          # Flask application
├── setup.py                        # Package installation configuration
├── requirements.txt                # Project dependencies
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Data loading and splitting
│   │   ├── data_transformation.py  # Feature engineering and preprocessing
│   │   └── model_trainer.py        # Model training and evaluation
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py       # Training pipeline
│   │   └── predict_pipeline.py     # Prediction pipeline
│   │
│   ├── exception.py                # Custom exception handling
│   ├── logger.py                   # Logging configuration
│   └── utils.py                    # Utility functions
│
├── artifacts/                      # Saved models and preprocessors
├── logs/                          # Application logs
├── notebook/                      # Jupyter notebooks for EDA
└── templates/                     # HTML templates for web interface
    ├── index.html
    └── home.html
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MLproject
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training pipeline to train all models and select the best one:
```bash
python src/pipeline/train_pipeline.py
```

### Running the Web Application

Start the Flask application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Making Predictions

1. Navigate to `http://localhost:5000` in your web browser
2. Click on "Predict Data"
3. Fill in the student information form:
   - Select gender
   - Choose race/ethnicity group
   - Select parental education level
   - Choose lunch type
   - Indicate test preparation course completion
   - Enter reading score (0-100)
   - Enter writing score (0-100)
4. Click submit to get the predicted math score

## Model Training Details

The project trains multiple regression models with the following hyperparameters:

- **Random Forest**: n_estimators tuning
- **Gradient Boosting**: learning_rate, subsample, n_estimators
- **XGBoost**: learning_rate, n_estimators
- **CatBoost**: depth, learning_rate, iterations
- **AdaBoost**: learning_rate, n_estimators
- **Decision Tree**: criterion tuning

The best performing model (based on R² score) is automatically selected and saved for predictions.

## Dependencies

- pandas
- numpy
- scikit-learn
- catboost
- xgboost
- Flask
- seaborn
- matplotlib
- dill

## Author

Mansi Sharma
- Email: sharmamansi1113@gmail.com

## License

This project is open source and available for educational purposes.
