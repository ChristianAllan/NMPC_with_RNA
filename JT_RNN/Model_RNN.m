function y = Model_RNN(net, X)

    % X: (features x time_steps)
    X = dlarray(X, 'CT');

    y = predict(net, X);
    y = extractdata(y);
end