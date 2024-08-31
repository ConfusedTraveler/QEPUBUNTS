import pandas as pd
import matplotlib.pyplot as plt
#import glob



# Extract unique N values

alpha_values = [1,15]

for a in alpha_values:
    # Step 2: Compute accuracy for AR1 model
    # Step 1: Read the pimax data
    if a == 1:
        x="b"
    else:
        x="a"
    pimax_file = 'Results/PIMAX/pimax_10000_'+str(a)+'.csv'
    pimax_data = pd.read_csv(pimax_file, header=None, names=['n', 'N', 'pimax'])
    N_values = pimax_data['N'].unique()
    ar1_accuracies = []
    for N in N_values:
        ar1_file = f'Results/AR1/ar1_pred_{N}_10000_{a}.csv'
        ar1_data = pd.read_csv(ar1_file, header=None, names=['X', 'pred', 'true'])
        ar1_accuracy = (ar1_data['pred'] == ar1_data['true']).mean()
        ar1_accuracies.append(ar1_accuracy)

    # Step 3: Compute accuracy for Markov model
    markov_accuracies = []
    for N in N_values:
        markov_file = f'Results/Markov/markov_pred_{N}_10000_{a}.csv'
        markov_data = pd.read_csv(markov_file, header=None, names=['X', 'pred', 'true'])
        markov_accuracy = (markov_data['pred'] == markov_data['true']).mean()
        markov_accuracies.append(markov_accuracy)

    # Step 4: Plot the results
    plt.figure(figsize=(8, 6))

    # Plot predictability upper bound
    plt.plot(pimax_data['N'], pimax_data['pimax'], label='Pimax', marker='o')

    # Plot AR1 accuracy
    plt.plot(N_values, ar1_accuracies, label='AR1', marker='s')

    # Plot Markov accuracy
    plt.plot(N_values, markov_accuracies, label='Markov', marker='^')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    # Customize the plot
    plt.xlabel('N',fontsize=20)
    plt.ylabel('Accuracy / Pimax',fontsize=25)
    plt.ylim(0,1)
    #plt.title('Predictability Upper Bound and Model Accuracies')
    plt.legend(fontsize=25)
    #plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"fig3{x}.png")

    # Show the plot
    plt.show()
