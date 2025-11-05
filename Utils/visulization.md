```mermaid
flowchart TD
    %% Data Layer
    subgraph "Data Layer"
        DS_temp["temperature.csv"]:::data
        DS_etth["ETTh1.csv"]:::data
        DS_stock["Stock_Open.csv"]:::data
        DS_markov_gen["generate_markov_synthetic.py"]:::data
        DS_syn1["Markov Exp1 CSVs"]:::data
        DS_syn2["Markov Exp2 CSVs"]:::data
    end

    %% Utility Layer
    subgraph "Utility Layer"
        U_gen["gen_data.py"]:::util
        U_H_exact["computeH_exact.py"]:::util
        U_PIMax["compute_PIMax.py"]:::util
        U_LZ1["LZ1.py"]:::util
        U_LZ2["LZ2.py"]:::util
    end

    %% Challenges Pipeline
    subgraph "Challenges Pipeline"
        C_fig2["fig2.py"]:::exp
        C_res(("24_50_fig2.csv")):::data
    end

    %% Experiment 1 Pipeline
    subgraph "Experiment 1 Pipeline"
        E1_pimax["pimax.py"]:::exp
        E1_ar1["AR1.py"]:::exp
        E1_markov["markov_predict.py"]:::exp
        E1_plot["plot_exp1.py"]:::plot
        E1_fig3["fig3.py"]:::plot
        E1_res1(("PIMAX Results")):::data
        E1_res2(("AR1 Results")):::data
        E1_res3(("Markov Results")):::data
    end

    %% Experiment 2 Pipeline
    subgraph "Experiment 2 Pipeline"
        E2_fig4a["exp1_fig2.py"]:::exp
        E2_fig4b["exp2_fig2.py"]:::exp
        E2_plot["plot_exp2_fig2.py"]:::plot
        E2_res(("Results")):::data
    end

    %% Experiment 3 Pipeline
    subgraph "Experiment 3 Pipeline"
        E3_temp["temp.py"]:::exp
        E3_etth["etth1.py"]:::exp
        E3_stock["stock.py"]:::exp
        E3_arima["arima_predict.py"]:::exp
        E3_lstm["lstm_predict.py"]:::exp
        E3_cnn["cnn_predict.py"]:::exp
        E3_conv["exp3_convergence.py"]:::exp
        E3_plot["plot_exp3.py"]:::plot
        E3_geo["pr_lon109_lat39.py"]:::exp
        E3_res_t(("Temp_Results")):::data
        E3_res_et(("ETTh1_Results")):::data
        E3_res_s(("Stock_Results")):::data
    end

    %% Data to Utils & Experiments
    DS_markov_gen --> U_gen
    U_gen --> DS_syn1
    U_gen --> DS_syn2
    DS_temp --> E3_temp
    DS_etth --> E3_etth
    DS_stock --> E3_stock
    DS_syn1 --> E1_pimax
    DS_syn1 --> C_fig2
    DS_syn2 --> E2_fig4a
    DS_syn2 --> E2_fig4b

    %% Utils usage
    U_H_exact -.-> E1_pimax
    U_PIMax -.-> E1_pimax
    U_LZ1 -.-> E1_pimax
    U_LZ2 -.-> E1_pimax
    U_PIMax -.-> C_fig2
    U_H_exact -.-> C_fig2
    U_PIMax -.-> E2_fig4a
    U_PIMax -.-> E2_fig4b
    U_H_exact -.-> E3_temp
    U_LZ1 -.-> E3_temp
    U_LZ2 -.-> E3_temp

    %% Challenges flow
    C_fig2 --> C_res

    %% Exp1 flow
    E1_pimax --> E1_res1
    E1_pimax --> E1_ar1
    E1_ar1 --> E1_res2
    E1_ar1 --> E1_markov
    E1_markov --> E1_res3
    E1_markov --> E1_plot
    E1_plot --> E1_fig3

    %% Exp2 flow
    E2_fig4a --> E2_plot
    E2_fig4b --> E2_plot
    E2_plot --> E2_res

    %% Exp3 flow
    E3_temp --> E3_arima
    E3_temp --> E3_lstm
    E3_temp --> E3_cnn
    E3_arima --> E3_res_t
    E3_lstm --> E3_res_t
    E3_cnn --> E3_res_t
    E3_etth --> E3_arima
    E3_etth --> E3_lstm
    E3_etth --> E3_cnn
    E3_arima --> E3_res_et
    E3_lstm --> E3_res_et
    E3_cnn --> E3_res_et
    E3_stock --> E3_arima
    E3_stock --> E3_lstm
    E3_stock --> E3_cnn
    E3_arima --> E3_res_s
    E3_lstm --> E3_res_s
    E3_cnn --> E3_res_s
    E3_res_t & E3_res_et & E3_res_s --> E3_conv
    E3_conv --> E3_plot
    E3_geo -.-> E3_conv

    %% Click Events
    click DS_temp "https://github.com/jamalsz/qepubunts/blob/main/Datasets/temperature.csv"
    click DS_etth "https://github.com/jamalsz/qepubunts/blob/main/Datasets/ETTh1.csv"
    click DS_stock "https://github.com/jamalsz/qepubunts/blob/main/Datasets/Stock_Open.csv"
    click DS_markov_gen "https://github.com/jamalsz/qepubunts/blob/main/Datasets/Markov/generate_markov_synthetic.py"
    click DS_syn1 "https://github.com/jamalsz/qepubunts/tree/main/Datasets/Markov/Exp1/"
    click DS_syn2 "https://github.com/jamalsz/qepubunts/tree/main/Datasets/Markov/Exp2/"
    click U_LZ1 "https://github.com/jamalsz/qepubunts/blob/main/Utils/LZ1.py"
    click U_LZ2 "https://github.com/jamalsz/qepubunts/blob/main/Utils/LZ2.py"
    click U_H_exact "https://github.com/jamalsz/qepubunts/blob/main/Utils/computeH_exact.py"
    click U_PIMax "https://github.com/jamalsz/qepubunts/blob/main/Utils/compute_PIMax.py"
    click U_gen "https://github.com/jamalsz/qepubunts/blob/main/Utils/gen_data.py"
    click C_fig2 "https://github.com/jamalsz/qepubunts/blob/main/Challenges_section/fig2.py"
    click C_res "https://github.com/jamalsz/qepubunts/blob/main/Challenges_section/Results/24_50_fig2.csv"
    click E1_pimax "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp1/pimax.py"
    click E1_ar1 "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp1/AR1.py"
    click E1_markov "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp1/markov_predict.py"
    click E1_plot "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp1/plot_exp1.py"
    click E1_fig3 "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp1/fig3.py"
    click E2_fig4a "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp2/exp1_fig2.py"
    click E2_fig4b "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp2/exp2_fig2.py"
    click E2_plot "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp2/plot_exp2_fig2.py"
    click E3_temp "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp3/temp.py"
    click E3_etth "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp3/etth1.py"
    click E3_stock "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp3/stock.py"
    click E3_arima "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp3/arima_predict.py"
    click E3_lstm "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp3/lstm_predict.py"
    click E3_cnn "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp3/cnn_predict.py"
    click E3_conv "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp3/exp3_convergence.py"
    click E3_plot "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp3/plot_exp3.py"
    click E3_geo "https://github.com/jamalsz/qepubunts/blob/main/Experiments/Exp3/pr_lon109_lat39.py"

    %% Styles
    classDef data fill:#E3F2FD,stroke:#1E88E5,stroke-width:1px;
    classDef util fill:#E8F5E9,stroke:#43A047,stroke-width:1px;
    classDef exp fill:#FFF3E0,stroke:#FB8C00,stroke-width:1px;
    classDef plot fill:#F3E5F5,stroke:#8E24AA,stroke-width:1px;