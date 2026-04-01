import pandas as pd


def data_preparation():
    # Read the previous campaign data
    df = pd.read_csv('../campaign_data.csv')
    
    # Clean the data if required (assuming no missing or incorrect values for this example)
    # For demonstration, let's just save it without changes
    df.to_csv('cleaned_campaign_data.csv', index=False)

data_preparation()

from sklearn.linear_model import LinearRegression


def model_training():
    # Read the cleaned data
    df = pd.read_csv('cleaned_campaign_data.csv')
    
    # Separate the predictors and response variable
    X = df['Marketing expenditure'].values.reshape(-1, 1)
    y = df['Units sold'].values
    
    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Save the model parameters (coefficients)
    # For linear regression, we save the intercept and coefficient
    parameters = {
        'intercept': model.intercept_,
        'coefficient': model.coef_[0]
    }
    
    pd.DataFrame([parameters]).to_csv('model_parameters.csv', index=False)

model_training()


def expenditure_calculation(desired_units_sold):
    # Read the model parameters
    params = pd.read_csv('model_parameters.csv')
    intercept = params['intercept'].values[0]
    coefficient = params['coefficient'].values[0]
    
    # Calculate the required marketing expenditure
    required_expenditure = (desired_units_sold - intercept) / coefficient
    
    # Save the result to a text file
    with open('required_expenditure.txt', 'w') as f:
        f.write(f"{required_expenditure:.2f}")

expenditure_calculation(60000)


def data_preparation():
    df = pd.read_csv('campaign_data.csv')
    df.to_csv('cleaned_campaign_data.csv', index=False)

def model_training():
    df = pd.read_csv('cleaned_campaign_data.csv')
    X = df['Marketing expenditure'].values.reshape(-1, 1)
    y = df['Units sold'].values
    model = LinearRegression()
    model.fit(X, y)
    parameters = {
        'intercept': model.intercept_,
        'coefficient': model.coef_[0]
    }
    pd.DataFrame([parameters]).to_csv('model_parameters.csv', index=False)

def expenditure_calculation(desired_units_sold):
    params = pd.read_csv('model_parameters.csv')
    intercept = params['intercept'].values[0]
    coefficient = params['coefficient'].values[0]
    required_expenditure = (desired_units_sold - intercept) / coefficient
    with open('required_expenditure.txt', 'w') as f:
        f.write(f"{required_expenditure:.2f}")

def desired_marketing_expenditure(marketing_expenditure, units_sold, desired_units_sold):
    data = {'Campaign': range(len(marketing_expenditure)),
            'Marketing expenditure': marketing_expenditure,
            'Units sold': units_sold}
    
    df = pd.DataFrame(data)
    df.to_csv('campaign_data.csv', index=False)
    
    data_preparation()
    model_training()
    expenditure_calculation(desired_units_sold)
    
    with open('required_expenditure.txt') as f:
        result = float(f.read())
    
    return result

# Example usage
print(desired_marketing_expenditure(
    [300000, 200000, 400000, 300000, 100000],
    [60000, 50000, 90000, 80000, 30000],
    60000))