import pandas as pd
import matplotlib.pyplot as plt

alpha_values = [1, 15]

for a in alpha_values:
    if a == 1:
        x = "b"
    else:
        x = "a"

    # Load Pimax data
    pimax_file = f'Results/PIMAX/pimax_5_50000_{a}.csv'
    pimax_data = pd.read_csv(pimax_file, header=None, names=['n', 'N', 'pimax'])
    N_values = pimax_data['N'].unique()

    # Compute AR1 accuracy
    ar1_accuracies = []
    for N in N_values:
        ar1_file = f'Results/AR1/ar1_pred_{N}_50000_{a}.csv'
        ar1_data = pd.read_csv(ar1_file, header=None, names=['X', 'pred', 'true'])
        ar1_accuracy = (ar1_data['pred'] == ar1_data['true']).mean()
        ar1_accuracies.append(ar1_accuracy)

    # Compute Markov accuracy
    markov_accuracies = []
    for N in N_values:
        markov_file = f'Results/Markov/markov_pred_{N}_50000_{a}.csv'
        markov_data = pd.read_csv(markov_file, header=None, names=['X', 'pred', 'true'])
        markov_accuracy = (markov_data['pred'] == markov_data['true']).mean()
        markov_accuracies.append(markov_accuracy)

    # Compute NN accuracy
    nn_accuracies = []
    for N in N_values:
        nn_file = f'Results/NN/nn_pred_{N}_50000_{a}.csv'
        nn_data = pd.read_csv(nn_file, header=None, names=['X', 'pred', 'true'])
        nn_accuracy = (nn_data['pred'] == nn_data['true']).mean()
        nn_accuracies.append(nn_accuracy)

    # Plot all accuracies
    plt.figure(figsize=(8, 6))

    plt.plot(N_values, pimax_data['pimax'], label='Pimax', marker='o')
    plt.plot(N_values, ar1_accuracies, label='AR1', marker='s')
    plt.plot(N_values, markov_accuracies, label='Markov', marker='^')
    plt.plot(N_values, nn_accuracies, label='NN', marker='x')

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('N', fontsize=20)
    plt.ylabel('Accuracy / Pimax', fontsize=25)
    plt.ylim(0, 1)
    plt.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig(f"fig3{x}.png")
    plt.show()
