import csv
import numpy as np
import pandas as pd
# import sys
import matplotlib.pyplot as plt
import warnings
import sys
sys.path.append('../')  # Add parent directory to the system path

from Utils.LZ2 import *
from Utils.LZ1 import *
from Utils.compute_PIMax import *

warnings.filterwarnings("ignore")

seed = 37

#Function to generate transition matrix
def generate_transition_matrix(N,p):

    """
    Generate a transition matrix of size NxN with a probability p of
    transitioning to a randomly chosen state, and a probability (1-p) of
    equiprobably transitioning to any of the other states

    Parameters
    ----------
    N : int
        Size of the transition matrix
    p : float
        Probability of transitioning to a randomly chosen state

    Returns
    -------
    P : array, shape (N, N)
        Transition matrix
    """
    P = np.zeros((N, N))
    np.random.seed(seed)

    for i in range(N):
        
        dominant_prob = 1 - (p*(N-1))  
        possible_next_states = [j for j in range(N) if j != i]
         # 从可能的下一个状态中随机选择一个作为主要转移目标，这个状态将获得最大的转移概率 dominant_prob
        dominant_next_state = np.random.choice(possible_next_states)
        P[i, dominant_next_state] = dominant_prob
        equiprobable_prob = p #(1 - dominant_prob) / (N - 1)
        
        for j in range(N):
            if j != dominant_next_state:
                P[i, j] = equiprobable_prob

    
    return P

# Function to train a Markov model and estimate the transition matrix
def train_markov_model(train_data, N):
    
    """
    Train a Markov model and estimate the transition matrix.
    
    Inside the function, a count matrix is initialized as a two-dimensional NumPy array with a shape of (N, N). 
    This matrix is intended to record the number of times the system transitions from one state to another. 
    As the function iterates through the train_data sequence, it treats each consecutive pair of elements as a state transition. 
    For every transition from a current state to its subsequent state, the corresponding entry in the count matrix is incremented.

    Once all transitions have been counted, the function converts the count matrix into a transition probability matrix. 
    This is done by dividing each count by the sum of counts in its corresponding row, 
    which results in each row summing to one (assuming there is at least one transition from that state). 
    The resulting matrix represents the probability of moving from one state to any other state in the model.
    
    Parameters
    ----------
    train_data : array_like
        The data to train the model on. This should be a sequence of states,
        where each state is an integer between 0 and N-1.
    N : int
        The number of states in the Markov model.

    Returns
    -------
    transition_matrix : array_like, shape (N, N)
        The estimated transition matrix of the Markov model.
    """
    np.random.seed(seed)
    count_matrix = np.zeros((N, N))
    for i in range(len(train_data) - 1):
        current_state = train_data[i]
        next_state = train_data[i + 1]
        count_matrix[int(current_state)][int(next_state)] += 1
    
    transition_matrix = count_matrix / count_matrix.sum(axis=1, keepdims=True)
    
    return transition_matrix

# Function to evaluate a Markov model using the transition matrix
def markov_model(eval_data, transition_matrix):
    """
    Evaluate a Markov model and generate predictions.

    Parameters
    ----------
    eval_data : array_like
        The data to evaluate the model on. This should be a sequence of states,
        where each state is an integer between 0 and N-1.
    transition_matrix : array_like, shape (N, N)
        The transition matrix of the Markov model.

    Returns
    -------
    predictions : array_like
        The predicted next states of the evaluation data.
    """
    N = len(transition_matrix)
    np.random.seed(seed)
    predictions = []
    
    for i in range(len(eval_data) - 1):
        current_state = eval_data[i]
        predicted_next_state = np.argmax(transition_matrix[int(current_state)])
        predictions.append(predicted_next_state)
    
    return np.array(predictions)

def predict(time_series, N):
    """
    Predict the next states of a time series using a Markov model.

    Parameters
    ----------
    time_series : array_like
        The time series to predict.
    N : int
        The number of states in the Markov model.

    Returns
    -------
    predictions : array_like
        The predicted next states of the time series.
    eval_data : array_like
        The evaluation data used to train the model.
    """
    np.random.seed(seed)
    n = len(time_series)
    train_size = int(0.8 * n)
    train_data, eval_data = time_series[:train_size], time_series[train_size:]
    transition_matrix = train_markov_model(train_data, N)
    predictions = markov_model(eval_data, transition_matrix)			
    return predictions,eval_data

def generate_sequence(transition_matrix, initial_state, sequence_length):
    """
    Generate a sequence of states using a Markov model.

    Parameters
    ----------
    transition_matrix : array_like, shape (N, N)
        The transition matrix of the Markov model.
    initial_state : int
        The initial state of the sequence.
    sequence_length : int
        The length of the sequence to generate.

    Returns
    -------
    sequence : array_like, shape (sequence_length,)
        The generated sequence of states.
    """
    current_state = initial_state
    sequence = [current_state]
    for _ in range(sequence_length - 1):
        # 根据转移矩阵中的概率分布随机选择下一个状态：
        # len(transition_matrix[current_state]) 提供可能的状态数量（0到N-1）
        # p=transition_matrix[current_state] 指定了从当前状态转移到每个可能状态的概率
        next_state = np.random.choice(len(transition_matrix[current_state]), p=transition_matrix[current_state])
        sequence.append(next_state)
        current_state = next_state
    return sequence	

def get_acc(predictions,eval_data):
    """
    Calculate the accuracy of a sequence of predictions against the evaluation data.

    Parameters
    ----------
    predictions : array_like
        The sequence of predictions.
    eval_data : array_like
        The evaluation data used to calculate the accuracy.

    Returns
    -------
    accuracy : float
        The accuracy of the predictions, calculated as the proportion of
        correctly predicted states.
    """
    p = 0
    for i in range(len(predictions)):
        if predictions[i]==eval_data[i+1]:
            p+=1		
    acc = p/len(predictions)			
    return acc			

def discretize(S,pred,eval_data,e):
	
    """
    Discretize the given sequences S, pred, and eval_data into a uniform grid of size N.
    
    Parameters
    ----------
    S : array_like
        The sequence to discretize.
    pred : array_like
        The sequence of predictions to discretize.
    eval_data : array_like
        The evaluation data to discretize.
    e : float
        The precision parameter for the discretization.
    
    Returns
    -------
    S_discretized : array_like
        The discretized sequence S.
    pred_discretized : array_like
        The discretized sequence of predictions.
    eval_discretized : array_like
        The discretized evaluation data.
    N : int
        The number of unique states in the discretized sequence.
    """
    N = int((max(S) - min(S) + 2 * e) / (2 * e))
    bins = np.linspace(min(S) - e, max(S) + e, N + 1)
    S_discretized = pd.cut(S, bins=bins, labels=False)
    pred_discretized = pd.cut(pred, bins=bins, labels=False)
    eval_discretized = pd.cut(eval_data, bins=bins, labels=False)
    N = len(np.unique(S_discretized))
    return S_discretized,pred_discretized,eval_discretized,N
	
if __name__ == "__main__":
    # 从命令行获取参数，first用于控制是生成新数据还是读取已有数据
    first = int(sys.argv[1])
    
    # 设置随机种子以确保结果可重现
    np.random.seed(seed)

    # 设置输出文件路径和离散化参数范围
    output_path = "Results/24_50_fig2.csv"
    eps = [1,2.5,5,7.5,10,12.5,15,17.5,20]  # 不同的离散化bin宽度

    if first == 1:
        # 生成新数据模式
        # 从命令行获取状态数N和序列长度n
        N = int(sys.argv[2])
        n = int(sys.argv[3])
        # 1.95是超参，设置的大于1，则转移概率小于1/N
        # dominant_prob = 1 - (p*(N-1))，向某个状态转移的概率最大，其他状态转移的概率相等，都是p
        p = 1/(N*1.95)  # 计算转移概率

        
        # 生成马尔可夫链转移矩阵和时间序列
        transition_matrix = generate_transition_matrix(N,p)
        initial_state = np.random.choice(N)
        series = generate_sequence(transition_matrix, initial_state, n)
        
        # 使用生成的序列进行预测
        tsize = int(0.8*len(series))  # 设置训练集大小
        predictions, eval_data = predict(series,N)
        
        # 存储不同bin宽度下的PImax和马尔可夫预测准确率
        pimax_l = []
        pimarkov = []
        
        # 对每个bin宽度进行计算
        for e in eps:
            print("eps:",e)
            # 对序列进行离散化处理
            S,pred,edata,NN = discretize(series,predictions,eval_data,e)
            # 计算LZ复杂度和PImax
            H = Compute_LZ2(S,0)
            pimax = get_pimax(H,NN)		
            # 计算预测准确率
            acc = get_acc(pred,edata)
            pimax_l.append(pimax)
            pimarkov.append(acc)
            
        # 保存结果到CSV文件
        csv_file = f'Results/{initial_state}_{N}_fig2.csv'
        data = list(zip(pimax_l, pimarkov))
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Pimax', 'Pimarkov'])
            writer.writerows(data)
    else:
         # 读取已有数据模式
         series = pd.read_csv(output_path)
         pimax_l = series['Pimax'].values
         pimarkov = series['Pimarkov'].values

    # 绘制结果图像
    plt.figure(figsize=(8, 5))	
    plt.plot(eps,pimax_l,marker= "o",label="Pimax")
    plt.plot(eps,pimarkov,marker="o",label="Pimarkov")
    plt.ylim(0.3,0.8)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Bin Width',fontsize=20)
    plt.ylabel('Accuracy/Pimax',fontsize=20)
    plt.legend(fontsize=16,loc='lower right')
    plt.tight_layout()
    # 保存并显示图像
    plt.savefig(f"./Results/fig2.png")
    plt.show()
    plt.close()
