import os
import fnmatch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_csv_files(N, alpha, dist):
    # Find all CSV files that match the given parameters
    length = [1000, 2000, 4000, 8000, 16000, 32000]
    file_pattern = [f"markov{ln}_{N}_{alpha}{dist}.csv" for ln in length]
    #matching_files = [file for file in os.listdir() if fnmatch.fnmatch(file, file_pattern)]
    #print(matching_files)

    # Read and concatenate the matching CSV files
    df_list = [pd.read_csv(file, header=None, ) for file in file_pattern]
    print(df_list[0])
    df = pd.concat(df_list, ignore_index=True)
    df.columns = ["n","eps","Pimax","H"]
    print(df.head())
    print(df.tail())

    return df

def calculate_ratio(df):
    # Assuming the dataframe has columns n, eps, Pimax, and H
    # Sort the dataframe based on n and eps
    df = df.sort_values(by=['n', 'eps'])
    
    print(df.head())


    # Calculate the ratio H_n_i / H_n_i-1
    df['H_ratio'] = df.groupby('eps')['H'].pct_change() + 1

    return df.pivot(index='eps', columns='n', values='H_ratio')

def plot_heatmap(data, N, alpha, dist):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data, annot=True, cmap='viridis', fmt=".2f", cbar_kws={'label': 'H_n_i / H_n_i-1'})
    plt.title(f"Heatmap for N={N}, alpha={alpha}, dist={dist}")
    plt.xlabel('n')
    plt.ylabel('eps')
    plt.show()

def main():
    # Set your parameters
    N = 50
    alpha = 0
    dist = 'u'

    # Read and concatenate CSV files
    df = read_csv_files(N, alpha, dist)

    # Calculate the ratio H_n_i / H_n_i-1
    heatmap_data = calculate_ratio(df)
    
    heatmap_data[heatmap_data.columns[0]] = 1
    

    # Plot the heatmap
    plot_heatmap(heatmap_data, N, alpha, dist)

if __name__ == "__main__":
    main()

