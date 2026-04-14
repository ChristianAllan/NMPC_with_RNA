function y = Func_RNN_Manual(X, W1, R1, b1, W2, R2, b2, W3, b3)

    T = size(X,2);
    
    h1 = zeros(64,1);
    h2 = zeros(32,1);

    % Simple RNN 1 (return_sequences = true)
    H1 = zeros(64,T);

    for t = 1:T
        h1 = tanh(W1*X(:,t) + R1*h1 + b1);
        H1(:,t) = h1;
    end

    % Simple RNN 2
    for t = 1:T
        h2 = tanh(W2*H1(:,t) + R2*h2 + b2);
    end

    % Dense
    y = W3*h2 + b3;
end