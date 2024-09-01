from flask import Flask, redirect, request, render_template, send_file
import pandas as pd
from io import BytesIO
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import pickle
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

app = Flask(__name__)

# Initialize global variables
df = pd.DataFrame()
trained_model = None
encoders = {}  # Dictionary to store encoders for each column
scalers = {}   # Dictionary to store scalers for each column
selected_features = []
trained_models = {}
target_column = ''
model_name = None
model_score = None


# Helper functions for preprocessing
def encode_categorical_columns(df):
    """Encode categorical columns using stored LabelEncoders."""
    for column in df.select_dtypes(include=['object', 'category']).columns:
        if column in encoders:
            df[column] = encoders[column].transform(df[column])
        else:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            encoders[column] = le  
    return df

def preprocess_data(df):
    """Preprocess data by encoding and scaling."""
    df = encode_categorical_columns(df)
    return df

def train_model(model_type):
    global df, selected_features, target_column, trained_models, model_score, trained_model
    
    try:
        X = df[selected_features]
        y = df[target_column]
        
        if X.isnull().any().any() or y.isnull().any():
            raise ValueError("Input data contains missing values.")
        
        X = preprocess_data(X)
        
        if model_type == 'LinearRegression':
            model = LinearRegression()
            model_name = 'Linear Regression'
        elif model_type == 'RandomForest':
            model = RandomForestRegressor()
            model_name = 'Random Forest'
        elif model_type == 'GradientBoosting':
            model = GradientBoostingRegressor()
            model_name = 'Gradient Boosting'
        elif model_type == 'DecisionTree':
            model = DecisionTreeRegressor()
            model_name = 'Decision Tree'
        elif model_type == 'SVR':
            model = SVR()
            model_name = 'Support Vector Regressor'
        elif model_type == 'KNeighbors':
            model = KNeighborsRegressor()
            model_name = 'K-Nearest Neighbors'
        elif model_type == 'AdaBoost':
            model = AdaBoostRegressor()
            model_name = 'AdaBoost Regressor'
        elif model_type == 'Lasso':
            model = Lasso()
            model_name = 'Lasso Regression'
        elif model_type == 'Ridge':
            model = Ridge()
            model_name = 'Ridge Regression'
        elif model_type == 'ElasticNet':
            model = ElasticNet()
            model_name = 'Elastic Net'
        elif model_type == 'ExtraTrees':
            model = ExtraTreesRegressor()
            model_name = 'Extra Trees Regressor'
        elif model_type == 'HistGradientBoosting':
            model = HistGradientBoostingRegressor()
            model_name = 'Histogram-Based Gradient Boosting'
        else:
            raise ValueError("Invalid model type selected.")
        
        model.fit(X, y)
        y_pred = model.predict(X)
        score = r2_score(y, y_pred)
        
        trained_models[model_name] = model
        trained_model = model  # Set the global trained_model
        
        return model_name, score
    
    except Exception as e:
        print(f"Exception during model training: {e}")
        return 'Error', None




def predict_example_input(example_input):
    global trained_models, selected_features, encoders, model_name
    
    try:
        if model_name is None or model_name not in trained_models:
            raise ValueError("No model has been trained or model is not available.")
        
        model = trained_models[model_name]
        
        # Convert example input into a DataFrame
        example_df = pd.DataFrame([example_input])
        
        # Ensure that only selected features are in the example input
        example_df = example_df[[col for col in selected_features if col in example_df.columns]]
        
        # Check for missing features in the example input
        missing_cols = set(selected_features) - set(example_df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")
        
        # Encode categorical data
        for column in example_df.columns:
            if column in encoders:
                if example_df[column].dtype == 'object':
                    if not set(example_df[column]).issubset(set(encoders[column].classes_)):
                        raise ValueError(f"Unexpected categorical values in input for column: {column}")
                    example_df[column] = encoders[column].transform(example_df[column])
        
        # Predict using the selected model
        prediction = model.predict(example_df[selected_features])
        
        return prediction[0], model_name
    
    except ValueError as e:
        print(f"ValueError during prediction: {e}")
        return f"Error in input data. Please ensure all inputs are valid and match the feature columns: {e}", model_name
    except KeyError as e:
        print(f"KeyError during prediction: {e}")
        return f"Input contains previously unseen labels. Please ensure all categorical inputs are valid: {e}", model_name
    except Exception as e:
        print(f"Exception during prediction: {e}")
        return f"Prediction error: {e}", model_name


def export_model():
    """Export the trained model as a pickle file."""
    global trained_model
    if trained_model:
        pickle_data = pickle.dumps(trained_model)
        file_object = BytesIO(pickle_data)
        return send_file(file_object, download_name='model.pkl', as_attachment=True, mimetype='application/octet-stream')
    return "No model is available for export."  # Handle the case where no model is available


def export_encoders_and_scalers():
    """Export encoders and scalers as a pickle file."""
    data = {'encoders': encoders, 'scalers': scalers}
    pickle_data = pickle.dumps(data)
    file_object = BytesIO(pickle_data)
    return send_file(file_object, download_name='encoders_scalers.pkl', as_attachment=True, mimetype='application/octet-stream')


@app.route('/user_manual')
def user_manual():
    return render_template('user_manual.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    global df, trained_models, selected_features, target_column, model_name, model_score
    
    prediction = None
    prediction_model = None  # To store the model name used for prediction
    columns = []
    model_score = None  # Initialize model_score here
    
    if request.method == 'POST':
        if 'upload_file' in request.files:
            file = request.files['upload_file']
            if file and file.filename.endswith('.csv'):
                try:
                    df = pd.read_csv(file)
                    print(f"Loaded dataframe with columns: {df.columns.tolist()}")
                except Exception as e:
                    print(f"Exception loading CSV file: {e}")
        
        if 'update_features_target' in request.form:
            selected_features = request.form.getlist('features')
            target_column = request.form.get('target')
            print(f"Selected features: {selected_features}")
            print(f"Target column: {target_column}")
        
        if 'train_model' in request.form:
            model_type = request.form.get('model_type')
            if model_type:
                model_name, score = train_model(model_type)
                model_score = score
                print(f"Model name: {model_name}, Model score: {model_score}")
        
        if 'example_input' in request.form:
            example_input = {}
            for col in selected_features:
                value = request.form.get(f'input_{col}')
                if value:
                    example_input[col] = value
            print(f"Example input: {example_input}")
            if model_name:
                prediction, prediction_model = predict_example_input(example_input)
                print(f"Prediction: {prediction}")
        
        if 'export_model' in request.form:
            response = export_model()
            if isinstance(response, str):  # Handle string responses (error messages)
                return response
            return response
        
        if 'export_encoders_scalers' in request.form:
            response = export_encoders_and_scalers()
            if response:
                return response
            return redirect(request.url)  # Redirect if no encoders/scalers are available
    
    columns = df.columns.tolist() if not df.empty else []

    return render_template('index.html',
                           columns=columns,
                           selected_features=selected_features,
                           target_column=target_column,
                           model_name=model_name,
                           model_score=model_score,
                           prediction=prediction,
                           prediction_model=prediction_model)




if __name__ == '__main__':
    app.run(debug=True)
