% Originally Created by H. Tankaria
% Last Modified by H. Tankaria on 2023-10-01
% This function preprocesses the Covtype dataset and returns a structured data object.
function data = COVTYPE(seed)
% COVTYPE - Preprocesses the Covtype dataset and returns a structured data object
%
% Syntax: data = COVTYPE(seed)
%
% Inputs:
%    seed - Random seed for reproducibility
%
% Outputs:
%    data - Struct containing normalized training and test data:
%           .x_train, .y_train, .x_test, .y_test, .w_init

    % Load dataset (assumed to include x_train, y_train, x_test, y_test)
    M = load('covtype.mat');
    
    % ---------------------
    % Preprocess Training Data
    % ---------------------
    [n, d] = size(M.x_train);
    
    % Normalize training features (zero mean, unit variance)
    D = M.x_train;
    s = std(D);
    s(s == 0) = 1;  % Avoid division by zero
    m = mean(D);
    D = (D - m) ./ s;

    % Add bias term (column of ones)
    D = [D, ones(n,1)];

    % Shuffle training data
    rng(seed);
    perm = randperm(n);
    A = D(perm, :);
    B = M.y_train(perm);

    % Store in output struct (transposed for column-wise access)
    data.x_train = A';
    data.y_train = B';

    fprintf('This is Covtype train data with n=%d, d=%d\n', size(data.x_train'));

    % ---------------------
    % Preprocess Test Data
    % ---------------------
    [nn, ~] = size(M.x_test);

    T = M.x_test;
    s = std(T);
    s(s == 0) = 1;
    m = mean(T);
    T = (T - m) ./ s;

    T = [T, ones(nn,1)];

    rng(seed);
    perm = randperm(nn);
    R = T(perm, :);
    S = M.y_test(perm);

    data.x_test = R';
    data.y_test = S';

    fprintf('This is Covtype test data with n=%d, d=%d\n', size(data.x_test'));

    % ---------------------
    % Initialize model weights
    % ---------------------
    rng(seed);
    data.w_init = randn(d + 1, 1);
end
