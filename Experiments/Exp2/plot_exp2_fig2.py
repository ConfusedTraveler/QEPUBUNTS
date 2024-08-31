import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_diff_scales(data):
    
    
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(data['P'].values[:], data['H_exact'].values[:],label='H_exact', marker='o', color="blue") 
    ax.plot(data['P'].values[:], data['H_LZ1'].values[:], label='H_LZ1', marker='o', color="green")
    ax.plot(data['P'].values[:], data['H_LZ2'].values[:], label='H_LZ2', marker='o',color="brown")
    ax.legend(["Hexact","HLZ1","HLZ2"])
    ax.set_yscale("linear")
    ax.set_xscale("log")
    ax.spines["bottom"].set_visible(False)  
    ax.get_xaxis().set_visible(False) 
    ax.set_ylabel("Entropy rate",fontsize=25)
    
    
    # Dividing the axes
    divider = make_axes_locatable(ax)
    ax_log = divider.append_axes("bottom", size=3, pad=0, sharex=ax)
    # Acting on the log axis
    ax_log.set_yscale("log")
    ax_log.set_xscale("log")
    ax_log.spines["top"].set_visible(False)  # hide top of box
    ax_log.plot(data['P'].values[:], data['H_exact'].values[:], marker='o', color="blue")
    ax_log.plot(data['P'].values[:], data['H_LZ1'].values[:],  marker='o', color="green")
    ax_log.plot(data['P'].values[:], data['H_LZ2'].values[:], marker='o',color="brown")
    ax_log.set_xlabel("Pw",fontsize=30)
    ax_log.set_ylabel("log axis",fontsize=30)
    ax_log.tick_params(labelsize=30)
    # Show delimiter
    ax.tick_params(labelsize=30)
    ax.legend(fontsize=30)
    ax.axhline(y=0, color="black", linestyle="dashed", linewidth=1)
    
    # Plotting proper
    fig.tight_layout()
    plt.savefig(f"fig4a.png")
    plt.show()

def plot_fig3b():
	file_path = 'Results/output_152_fig2.txt' #'Results/output_exp1_' + str(n)+'_0u.txt'  
	column_names = ['N', 'H_LZ1', 'H_LZ2', 'H_exact']
	data = pd.read_csv(file_path, header=None, names=column_names)

	data = data.sort_values(by='N')

	# Plot the line graph
	plt.figure(figsize=(10, 8))

	# Plot Pi_max_exact
	plt.plot(data['N'], data['H_exact'], label='H_exact', marker='o', color="blue")

	# Plot Pi_max_LZ1
	plt.plot(data['N'], data['H_LZ1'], label='H_LZ1', marker='o', color="green")

	# Plot Pi_max_LZ2
	plt.plot(data['N'], data['H_LZ2'], label='H_LZ2', marker='o',color="brown")



	#plt.ylim(0,1)

	# Set labels and title
	plt.xticks(fontsize=30)
	plt.yticks(fontsize=30)
	plt.xlabel('N', fontsize=30)
	plt.ylabel('Entropy Rate',fontsize=30)
	#plt.title(f'n= {n}')
	plt.legend(fontsize=30)
	#plt.grid(True)
	plt.tight_layout()

	plt.savefig(f"fig4b.png")
	plt.show()
	plt.close()

if __name__ == "__main__":
	
	file_fig4a = './Results/exp2_fig2_10000_10.txt'
	column_names = ['N', 'P', 'H_LZ1', 'H_LZ2', 'H_exact']
	data = pd.read_csv(file_fig4a, header=None, names=column_names)
	
	
	n=10000
	plot_diff_scales(data)
	plot_fig3b()
	

	
	'''
      
    # Plot the line graph
	plt.figure(figsize=(10, 6))
	
	# Plot Pi_max_exact linear scale
	plt.plot(data['P'].values, data['H_exact'].values, label='H_exact', marker='o', color="blue")
	
	# Plot Pi_max_LZ1
	plt.plot(data['P'].values, data['H_LZ1'].values, label='H_LZ1', marker='o', color="green")
	
	# Plot Pi_max_LZ2
	plt.plot(data['P'].values, data['H_LZ2'].values, label='H_LZ2', marker='o',color="brown")
	
	#plt.plot(data['P'].values,np.ones(len(data['P'].values))*data['H_exact'].values[x1:x1+1],"--")
	
	# Plot Pi_max_exact linear scale
	plt.semilogy(data['P'].values, data['H_exact'].values, marker='o', color="blue")
	
	
	# Plot Pi_max_LZ1
	plt.semilogy(data['P'].values, data['H_LZ1'].values,  marker='o', color="green")
	
	
	# Plot Pi_max_LZ2
	plt.semilogy(data['P'].values, data['H_LZ2'].values, marker='o',color="brown")
	
	plt.xscale('log')
	
	#plt.xticks(fontsize=16)
	#plt.yticks(fontsize=16)
	#plt.ylim(0,1)

	# Set labels and title
	plt.xlabel('p')
	plt.ylabel('Entropy Rate')
	#plt.title(f'n= {n}')
	plt.legend()
	#plt.grid(True)
	
	plt.savefig(f"fig4a2.png")
	plt.show()
	plt.close()
	# Plot the line graph
	plt.figure(figsize=(10, 6))
	
	# Plot Pi_max_exact
	plt.plot(data['P'], data['Pi_max_Exact'], label='Pi_max_Exact', marker='o', color="blue")
	
	# Plot Pi_max_LZ1
	plt.plot(data['P'], data['Pi_max_LZ1'], label='Pi_max_LZ1', marker='o', color="green")
	
	# Plot Pi_max_LZ2
	plt.plot(data['P'], data['Pi_max_LZ2'], label='Pi_max_LZ2', marker='o',color="brown")
	
	# Plot Pi_max_markov
	plt.plot(data['P'], data['Pi_Markov'], label='Pi_Markov', marker='o', color="red")
	
	#plt.ylim(0,1)
	plt.xscale('log')

	# Set labels and title
	plt.xlabel('N',fontsize=16)
	plt.ylabel('Predictability',fontsize=16)
	#plt.title(f'n= {n}')
	plt.legend(fontsize=16)
	plt.grid(True)
	
	plt.savefig(f"fig4a3.png")
	plt.show()
	plt.close()
      
    '''

		

