function data = ADULT(seed)
% ADULT - Loads and preprocesses the UCI Adult dataset for training and testing
% Input:
%   seed - Random seed for reproducibility
% Output:
%   data - Struct containing normalized training and test sets with bias term, and random init weights

    % Load pre-saved dataset (x_train, y_train, x_test, y_test)
    M = load('adult.mat'); 

    % ------------------ Preprocess Training Data ------------------
    [n, d] = size(M.x_train);       % Number of samples and features
    D = M.x_train;

    % Normalize features to zero mean and unit variance
    s = std(D);
    s(s == 0) = 1;                  % Prevent division by zero
    m = mean(D);
    D = (D - m) ./ s;

    % Add bias term (column of ones)
    D = [D, ones(n,1)];

    % Shuffle training data using seed
    rng(seed);
    perm = randperm(n);
    A = D(perm,:);
    B = M.y_train(perm);

    % Assign to data struct
    data.x_train = A';
    data.y_train = B';
    fprintf('This is Adult train data with n=%d, d=%d\n', size(data.x_train'));

    % ------------------ Preprocess Test Data ------------------
    P = M.x_test;
    [e, ~] = size(P);

    % Normalize test features
    s = std(P);
    s(s == 0) = 1;
    m = mean(P);
    P = (P - m) ./ s;

    % Add bias term and shuffle
    P = [P, ones(e,1)];
    rng(seed);
    per = randperm(e);
    data.x_test = P(per,:)';
    data.y_test = M.y_test(per)';
    fprintf('This is Adult test data with n=%d, d=%d\n', size(data.x_test'));

    % ------------------ Initialize Random Weights ------------------
    rng(seed);
    data.w_init = randn(d + 1, 1);

end
% Note: The dataset 'adult.mat' should contain the variables x_train, y_train, x_test, and y_test.