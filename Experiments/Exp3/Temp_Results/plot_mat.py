import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Define the list of lengths
if __name__ == "__main__":
	lengths = [100,500,1000,2000,4000,6000,8000,10000,12000]
	# Read the CSV file with no headers and column names
	column_names = ["n", "eps", "pimax"]
	data = pd.read_csv("n_e.csv", names=column_names, header=None)
	data["piarima"] = -1.0
	ylabels = sorted(data["eps"].unique())
	# Initialize a variable to keep track of the correct predictions

	print(data.head())
	# Iterate over the lengths and process the corresponding CSV file
	for n in lengths:
		file_name = f"arima_temp_n{n}_e.csv"  

		with open(file_name, 'r') as csv_file:
			csv_reader = csv.reader(csv_file)
			total_predictions = 0
			# Skip the header row if present
			cpred = {}
			for e in ylabels:
				cpred[e]=0
			for row in csv_reader:
				total_predictions += 1
				for e in ylabels:
					N, pred, true = map(float, row)
					if abs(pred - true) <= e:
						cpred[e] += 1
			# Calculate accuracy
			print(n,e)
			for e in ylabels:
				accuracy = cpred[e]/total_predictions
				print(accuracy)
				data.loc[(data["n"]==n) & (data["eps"]==e),"piarima"] = accuracy



	data = data[data["piarima"] != -1]

	# Get unique values of 'n' and 'eps'
	xlabels = sorted(data["n"].unique())


	# Create a pivot table for the heatmap
	heatmap_data = (data.pivot("eps", "n", "pimax") - data.pivot("eps", "n", "piarima")).values
	print(data.pivot("eps", "n", "pimax") - data.pivot("eps", "n", "piarima"))
	# Create the figure and axes
	fig, ax = plt.subplots()

	# Create the heatmap
	im = ax.imshow(heatmap_data,cmap="RdYlGn",aspect='auto')

	# Set axis labels and titles
	ax.set_xticks(np.arange(len(xlabels)))
	ax.set_yticks(np.arange(len(ylabels)))
	ax.set_xticklabels(xlabels)
	ax.set_yticklabels(ylabels)

	# Rotate the tick labels and set their alignment
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	for i in range(len(ylabels)):
		for j in range(len(xlabels)):
		    value = np.around(heatmap_data[i, j], decimals=2)
		    text = ax.text(j, i, str(value), ha="center", va="center", color="black")

	# Save the plot as a .png file
	plt.savefig("heatmap.png")

	# Display the plot
	plt.show()



