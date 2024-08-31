import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import os

def get_data(path, length):
    df = pd.read_csv(path)
    df = df.iloc[:length, :]
    return df.iloc[:, 0].values 

# Load time series data from CSV
#data = pd.read_csv('time_series_data.csv', header=None, names=['Value'])

def model_train_pred(data,output_path,N):
    # Split data into training and test sets
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    # Train ARIMA(1,0,0) model (AR(1))
    model = ARIMA(train_data, order=(1,0,0))
    model_fit = model.fit()

    # Make predictions on test data
    predictions = model_fit.forecast(steps=len(test_data))

    # Save true values and predicted values into a CSV file
    
    #results = pd.DataFrame({'True_Values': test_data['Value'], 'Predicted_Values': predictions}, index=test_data.index)
    #results.to_csv(output_path)

    with open(output_path,"a") as f:
        for j in range(len(predictions)):
            f.write(str(N)+","+str(round(predictions[j]))+","+str(test_data[j])+"\n")

    print("True and predicted values saved to ",output_path)

    return

if __name__ == "__main__":

    N_values = [5,10,15,20,25,30,35,40,45,50]
    n = 10000
    a_values = [1,15]

    # Get the current directory of the script
    current_directory = os.getcwd()
    
    # Navigate to the parent directory
    parent_directory = os.path.dirname(current_directory)
    
    # Navigate to the parent of the parent directory (grandparent directory)
    grandparent_directory = os.path.dirname(parent_directory)
    for a in a_values:
         for N in N_values:
            input_path = grandparent_directory + "/Datasets/Markov/Seed11/markov_ts_"+str(N)+"_"+str(50000)+"_"+str(a)+"z.csv"
            output_path = "Results/ar1_pred_"+str(N)+"_"+str(n)+"_"+str(a)+".csv"
            series = get_data(input_path, n)
            model_train_pred(series,output_path)
            

