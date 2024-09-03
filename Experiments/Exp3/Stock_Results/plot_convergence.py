import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_pivot(df):

	res = df.copy()
	
	for i in range(len(df.index)):
		for j in range(1,len(df.columns)):
			res.iloc[i,j] = abs(df.iloc[i,j] - df.iloc[i,j-1])/max(df.iloc[i,j-1],df.iloc[i,j-1])
			
	return res
def calculate_ratio(dft,cl):
    # Assuming the dataframe has columns n, eps, Pimax, and H
    # Sort the dataframe based on n and eps
    df = dft.copy()
    df = df.sort_values(by=['n', 'eps'])
    
    df = df.pivot(index='eps', columns='n', values=cl) 
    print(df)
    #df = df.drop(250, axis=1)
    #df = df.drop(500, axis=1)
      
    pivot = get_pivot(df)
    print(pivot)
    #pivot = pivot.drop(500, axis=1)

    return pivot

if __name__ == "__main__":
	#lengths = [100,500,1000,2000,4000,6000,8000,10000,12000]
	# Read the CSV file with no headers and column names
	column_names = ["n", "eps", "HLZ1","HLZ2"]

	# Initialize an empty dataframe to store the concatenated data
	
	
	# Iterate through each CSV file and append its data to the combined dataframe

	file_path = "stock_conv.csv"
	data = pd.read_csv(file_path, header=None)
	data.columns = column_names
	


	# Create a pivot table for the heatmap
	heatmap_data = calculate_ratio(data,"HLZ1")#(data.pivot("eps", "n", "HLZ1")).values
	heatmap_data = heatmap_data.drop(1000,axis=1)
	print(heatmap_data)
	ax = sns.heatmap(heatmap_data, annot=False, fmt=".2f", cmap="YlGnBu",cbar=False)
	for i in range(heatmap_data.shape[0]):
		for j in range(heatmap_data.shape[1]):
			ax.text(j + 0.5, i + 0.5, np.around(heatmap_data.iloc[i, j],decimals=2), ha="center", va="center", color="red")
	

	# Customize other plot properties if needed
	#plt.title(f'N = {N} Convergence threshold=1%')
	plt.xlabel('n')
	plt.ylabel('eps')

	# Save the plot as a .png file
	plt.savefig("heatmap_stock_HLZ1.png")

	# Display the plot
	plt.show()
	
	# Create a pivot table for the heatmap
	heatmap_data = calculate_ratio(data,"HLZ2")#(data.pivot("eps", "n", "HLZ1")).values
	heatmap_data = heatmap_data.drop(1000,axis=1)
	print(heatmap_data)
	ax = sns.heatmap(heatmap_data, annot=False, fmt=".2f", cmap="YlGnBu",cbar=False)
	for i in range(heatmap_data.shape[0]):
		for j in range(heatmap_data.shape[1]):
			ax.text(j + 0.5, i + 0.5, np.around(heatmap_data.iloc[i, j],decimals=2), ha="center", va="center", color="red")
	

	# Customize other plot properties if needed
	#plt.title(f'N = {N} Convergence threshold=1%')
	plt.xlabel('n')
	plt.ylabel('eps')

	# Save the plot as a .png file
	plt.savefig("heatmap_stock_HLZ2.png")

	# Display the plot
	plt.show()



