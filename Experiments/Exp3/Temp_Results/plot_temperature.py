import pandas as pd
import matplotlib.pyplot as plt

def get_pi_model(pred,actual,eps):
	acc = []
	for e in eps:
		p = 0
		for i in range(len(pred)):
			if abs(pred[i] - actual[i]) <= e:
				p+=1
		acc.append(p/len(pred))
	return acc


df_arima = pd.read_csv('arima_temperature.csv', names=['length','Actual','Pred'], header=None)
df_cnnlstm = pd.read_csv('cnnlstm_temperature.csv')#, names=['Actual','Pred'])
df_scinet = pd.read_csv('scinet_temperature.csv')

df_lstm = pd.read_csv('lstm_temperature.csv')#, names=['Actual','Pred'])

# Read the CSV file into a DataFrame and manually assign column names
column_names = ['length', 'epsilon', 'pimax','H']
dfk = pd.read_csv('pimax_temp_KY.csv', names=column_names, header=None)

dfcn = pd.read_csv('pimax_temp_Hcn.csv', names=column_names, header=None)

#dfk = dfk[dfk['length'] == 35000 ]
dfk = dfk.sort_values(by='epsilon')
dfcn = dfcn.sort_values(by='epsilon')
# Get distinct epsilon values and sort them
eps = sorted(dfk['epsilon'].unique())
arima_acc = get_pi_model(df_arima['Pred'].values,df_arima['Actual'].values,eps)
scinet_acc = get_pi_model(df_scinet['Pred'].values,df_scinet['True'].values,eps)
cnn_acc = get_pi_model(df_cnnlstm['Predicted'].values,df_cnnlstm['Actual'].values,eps)
lstm_acc = get_pi_model(df_lstm['Predicted'].values,df_lstm['Actual'].values,eps)
# Create a plot for Pimaxl, Pimax, and Pimaxh
plt.figure(figsize=(8, 6))
plt.plot(eps, dfk['pimax'].values,  "-o",color="blue")
plt.plot(eps, dfcn['pimax'].values, "-o",color="green")
plt.plot(eps, scinet_acc, "-o",color="magenta")
plt.plot(eps, arima_acc, "-o", color="red")
plt.plot(eps, lstm_acc,"-o",color="orange")
plt.plot(eps, cnn_acc,"-o",color="purple")
#plt.title(f'Pimax vs Pimodel')
plt.xlabel('Epsilon',fontsize=22)
plt.ylabel('Predictability',fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.ylim(0,1)
plt.legend(["Pimax_LZ2","Pimax_LZ1","Pi_Scinet","Pi_arima","Pi_lstm","Pi_cnnlstm"],fontsize=22)
plt.tight_layout()
plt.savefig(f'Temp_Pimax_Pimodel_cn_ky.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(eps, dfk['H'].values,  "-o", color="blue")
plt.plot(eps, dfcn['H'].values, "-o", color="green")
plt.xlabel('Epsilon',fontsize=22)
plt.ylabel('Entropy Rate',fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.tight_layout()
#plt.ylim(0,1)
plt.legend(["H_LZ2","H_LZ1"],fontsize=22)
plt.savefig(f'Temp_H_cn_ky.png')
plt.close()















