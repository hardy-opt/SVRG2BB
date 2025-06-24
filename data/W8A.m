function data = W8A(seed)
% W8A - Preprocesses the W8A dataset and returns a structured data object
%
% Syntax: data = W8A(seed)
%
% Inputs:
%    seed - Random seed for reproducibility
%
% Outputs:
%    data - Struct containing normalized and shuffled data:
%           .x_train, .y_train, .x_test, .y_test, .w_init

    % Load dataset (should include x_train, y_train, x_test, y_test)
    M = load('w8a.mat');

    % ---------------------
    % Preprocess Training Data
    % ---------------------
    D = M.x_train;
    [n, d] = size(D);

    % Normalize features (zero mean, unit variance)
    s = std(D);
    s(s == 0) = 1;  % Avoid division by zero
    m = mean(D);
    D = (D - m) ./ s;

    % Add bias term (column of ones)
    D = [D, ones(n, 1)];

    % Shuffle training data
    rng(seed);
    perm = randperm(n);
    A = D(perm, :);
    B = M.y_train(perm);

    % Store in output struct (transposed)
    data.x_train = A';
    data.y_train = B';

    fprintf('This is W8A train data with n=%d, d=%d\n', size(data.x_train'));

    % ---------------------
    % Preprocess Test Data
    % ---------------------
    T = M.x_test;
    [nt, ~] = size(T);

    % Shuffle test data and add bias term
    rng(seed);
    perm = randperm(nt);
    T = T(perm, :);
    T = [T, ones(nt, 1)];

    data.x_test = T';
    data.y_test = M.y_test';

    fprintf('This is W8A test data with n=%d, d=%d\n', size(data.x_test'));

    % ---------------------
    % Initialize model weights
    % ---------------------
    rng(seed);
    data.w_init = randn(d + 1, 1);

end
