clc

% Camadas e Pesos

% ===== RNN 1 =====
W1 = readmatrix('Pesos/peso_0.csv'); % (2,64)
R1 = readmatrix('Pesos/peso_1.csv'); % (64,64)
b1 = readmatrix('Pesos/peso_2.csv'); % (64,)
b1 = b1(:);

% ===== RNN 2 =====
W2 = readmatrix('Pesos/peso_3.csv'); % (64,32)
R2 = readmatrix('Pesos/peso_4.csv'); % (32,32)
b2 = readmatrix('Pesos/peso_5.csv'); % (32,)
b2 = b2(:);

% ===== Dense =====
W3 = readmatrix('Pesos/peso_6.csv'); % (32,2)
b3 = readmatrix('Pesos/peso_7.csv'); % (2,)
b3 = b3(:);

W1 = W1';   % (64,2)
%R1 = R1';   % (64,64)

W2 = W2';   % (32,64)
%R2 = R2';   % (32,32)

W3 = W3';   % (2,32)

%% Entrada Degrau

%T = 50;
%u1 = [zeros(1,25), 6*ones(1,25)]; % degrau
%u2 = 0.2*ones(1,T);
fi = 15;
fv = 15;
u1 = [fi*ones(1,6)];
u2 = [fv*ones(1,6)];
X = [u1; u2];

% Normalizar entrada
meanX = [11.54549372, 12.4514907]';
stdX = [4.23449211, 3.9104502]';
meanX = meanX(:);
stdX  = stdX(:);

Xn = Standard_Norm(X, meanX, stdX);

% Normalizar saida
meanY = [3.07519347, 46.57397573]';
stdY = [2.11485258, 3.02168235]';
meanY = meanY(:);
stdY  = stdY(:);

%% X = [features x time_steps]
%y = Func_RNN_Manual(Xn, W1, R1, b1, W2, R2, b2, W3, b3);

y = Func_Manual(Xn, W1, R1, b1, W2, R2, b2, W3, b3);
Y = Standard_Desnorm(y, meanY, stdY)