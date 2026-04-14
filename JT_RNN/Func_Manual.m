function w = Func_Manual(X, W1, U1, b1, W2, U2, b2, W3, b3)
    
    % --- PROCESSAMENTO CAMADA 1 (return_sequences=True) ---
    h1 = zeros(64, 1); % Estado oculto inicial
    out_seq1 = zeros(64, 6); 
    for t = 1:6
        h1 = tanh(W1 * X(:, t) + U1 * h1 + b1);
        out_seq1(:, t) = h1;
    end
    
    % --- PROCESSAMENTO CAMADA 2 (return_sequences=False / last) ---
    h2 = zeros(32, 1);
    for t = 1:6
        h2 = tanh(W2 * out_seq1(:, t) + U2 * h2 + b2);
    end
    % h2 agora é o vetor do último passo (Last)
    
    % --- CAMADA DENSE FINAL ---
    w = W3 * h2 + b3;