function data = MNIST38(seed)
% MNIST38 - Preprocesses the MNIST 3-vs-8 dataset and returns a structured data object
%
% Syntax: data = MNIST38(seed)
%
% Inputs:
%    sed - Seed offset for reproducibility
%
% Outputs:
%    data - Struct containing normalized and shuffled data:
%           .x_train, .y_train, .x_test, .y_test, .w_init

    % Adjusted seed to ensure different values across calls
    seed = sed + 1;

    % Load dataset (assumed to contain x_train, y_train, x_test, y_test)
    M = load('MNIST38.mat');

    % ---------------------
    % Preprocess Training Data
    % ---------------------
    D = M.x_train;
    [n, d] = size(D);

    % Normalize features (zero mean, unit variance)
    s = std(D);
    s(s == 0) = 1;
    m = mean(D);
    D = (D - m) ./ s;

    % Add bias term (column of ones)
    D = [D, ones(n, 1)];

    % Shuffle training data
    rng(seed);
    perm = randperm(n);
    A = D(perm, :);
    B = M.y_train(perm);

    % Store in output struct (transposed for column-wise access)
    data.x_train = A';
    data.y_train = B';

    fprintf('This is MNIST38 train data with n=%d, d=%d\n', size(data.x_train'));

    % ---------------------
    % Preprocess Test Data
    % ---------------------
    T = M.x_test;
    [nt, ~] = size(T);

    % Shuffle and add bias term directly (without normalization for test set)
    rng(seed);
    perm = randperm(nt);
    T = T(perm, :);
    T = [T, ones(nt, 1)];

    data.x_test = T';
    data.y_test = M.y_test';

    fprintf('This is MNIST38 test data with n=%d, d=%d\n', size(data.x_test'));

    % ---------------------
    % Initialize model weights
    % ---------------------
    rng(seed);
    data.w_init = randn(d + 1, 1);

end
