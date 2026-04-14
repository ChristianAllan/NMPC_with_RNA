clc
clear all

% Pesos
% ===== RNN 1 =====
W1 = readmatrix('Pesos/peso_0.csv')'; % (2,64)
R1 = readmatrix('Pesos/peso_1.csv'); % (64,64)
b1 = readmatrix('Pesos/peso_2.csv'); % (64,)
b1 = b1(:);

% ===== RNN 2 =====
W2 = readmatrix('Pesos/peso_3.csv')'; % (64,32)
R2 = readmatrix('Pesos/peso_4.csv'); % (32,32)
b2 = readmatrix('Pesos/peso_5.csv'); % (32,)
b2 = b2(:);

% ===== Dense =====
W3 = readmatrix('Pesos/peso_6.csv')'; % (32,2)
b3 = readmatrix('Pesos/peso_7.csv'); % (2,)
b3 = b3(:);

%% Teste

% INPUT_SIZE = 2, TIME_STEPS = 6, OUTPUT_SIZE = 2
layers = [
    sequenceInputLayer(2, 'Name', 'input')
    
    % Se a fullyConnectedRecurrentLayer falha, tente a unitária:
    layertool.simpleRNNLayer(64, 'OutputMode', 'sequence', 'Name', 'rnn1')
    
    layertool.simpleRNNLayer(32, 'OutputMode', 'last', 'Name', 'rnn2')
    
    fullyConnectedLayer(2, 'Name', 'fc_out')
    regressionLayer('Name', 'output')
];

%% 1. Definição da Arquitetura

layers = [
    sequenceInputLayer(2, 'Name', 'input') % INPUT_SIZE = 2
    % 'sequence' -> retorna sequência completa
    % 'last' -> retorna apenas resultado final
    fullyConnectedRecurrentLayer(64, 'OutputMode', 'sequence', 'Name', 'rnn1')
    
    fullyConnectedRecurrentLayer(32, 'OutputMode', 'last', 'Name', 'rnn2')
    
    fullyConnectedLayer(2, 'Name', 'fc_out') % OUTPUT_SIZE = 2
    regressionLayer('Name', 'output')
];


%% 2. Atribuição dos Pesos Manualmente

% Camada RNN 1 (Índice 2)
layers(2).InputWeights = W1;
layers(2).RecurrentWeights = U1;
layers(2).Bias = b1;

% Camada RNN 2 (Índice 3)
layers(3).InputWeights = W2;
layers(3).RecurrentWeights = U2;
layers(3).Bias = b2;

% Camada Dense (Índice 4)
layers(4).Weights = W3;
layers(4).Bias = b3;

% 3. Montagem da Rede
net = assembleNetwork(layers);

% Exemplo: 2 variáveis de entrada em 6 passos de tempo
% O dado deve ter formato [INPUT_SIZE x TIME_STEPS] -> [2 x 6]
dados_entrada = rand(2, 6); 

% Se você tiver várias amostras (ex: 10 amostras), use uma Cell Array
% entrada_batch = {rand(2,6), rand(2,6), ..., rand(2,6)};

resultado = predict(net, dados_entrada);
disp(resultado); % Saída será [2 x 1] (o último passo predito para as 2 saídas)