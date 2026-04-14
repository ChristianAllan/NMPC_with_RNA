function X_norm = Standard_Norm(X, meanX, stdX)
    X_norm = (X - meanX) ./ stdX;
end