clc

INPUT_SIZE = 2;
OUTPUT_SIZE = 2;

layers = [
    sequenceInputLayer(INPUT_SIZE)

    recurrentLayer(64)
    recurrentLayer(32)

    fullyConnectedLayer(OUTPUT_SIZE)
];

net = dlnetwork(layers);

% ===== RNN 1 =====
W1 = readmatrix('peso_0.csv'); % (2,64)
R1 = readmatrix('peso_1.csv'); % (64,64)
b1 = readmatrix('peso_2.csv'); % (64,)

% ===== RNN 2 =====
W2 = readmatrix('peso_3.csv'); % (64,32)
R2 = readmatrix('peso_4.csv'); % (32,32)
b2 = readmatrix('peso_5.csv'); % (32,)

% ===== Dense =====
W3 = readmatrix('peso_6.csv'); % (32,2)
b3 = readmatrix('peso_7.csv'); % (2,)

W1 = W1';   % (64,2)
R1 = R1';   % (64,64)
b1 = b1(:);

W2 = W2';   % (32,64)
R2 = R2';   % (32,32)
b2 = b2(:);

W3 = W3';   % (2,32)
b3 = b3(:);

% RNN 1
net.Layers(2).InputWeights = W1;
net.Layers(2).RecurrentWeights = R1;
net.Layers(2).Bias = b1;

% RNN 2
net.Layers(3).InputWeights = W2;
net.Layers(3).RecurrentWeights = R2;
net.Layers(3).Bias = b2;

% Dense
net.Layers(4).Weights = W3;
net.Layers(4).Bias = b3;