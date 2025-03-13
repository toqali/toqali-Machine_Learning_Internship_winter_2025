# Taxi Fare Prediction

This project involves predicting taxi fares using a machine learning model and providing a user interface for interaction. The project utilizes Django for the web framework, LightGBM Regressor for the machine learning model, and StandardScaler for data preprocessing.

## Project Structure

- **README.md**: Project documentation.
- **src**: Source code for the project.
    - **models**: Contains the machine learning models.
    - **webapp**: Django web application.

## Technologies Used

- **Django**: A high-level Python web framework that encourages rapid development and clean, pragmatic design.
- **LightGBM Regressor**: A gradient boosting framework that uses tree-based learning algorithms for regression tasks.
- **StandardScaler**: A preprocessing technique to standardize the features by removing the mean and scaling to unit variance.

## Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone <https://github.com/toqali/toqali-Machine_Learning_Internship_winter_2025/tree/Internship_1/Task6/FareSpot_OmarMamon>
    cd <FareSpot_OmarMamon>
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run migrations**:
    ```bash
    python manage.py migrate
    ```

4. **Start the Django server**:
    ```bash
    python manage.py runserver
    ```

## Usage

- Access the web application at `http://127.0.0.1:8000/`.
- Input the required features to get the predicted taxi fare.

## Model Training

1. **Data Preprocessing**:
    - StandardScaler is used to standardize the dataset.

2. **Model Training**:
    - LightGBM Regressor is used to train the model on the preprocessed data.

## Model Performance

- **R² Score**: 0.7846
- **MSE**: 0.0429
- **RMSE**: 0.2070
- **MAE**: 0.1447

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Contact
For any inquiries, please contact [here](mailto:omar.mamon203@gmail.com).
