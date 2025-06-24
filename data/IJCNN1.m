function data = IJCNN1(seed)
% IJCNN1 - Preprocesses the IJCNN1 dataset and returns a structured data object
%
% Syntax: data = IJCNN1(seed)
%
% Inputs:
%    seed - Random seed for reproducibility
%
% Outputs:
%    data - Struct containing normalized and shuffled data:
%           .x_train, .y_train, .x_test, .y_test, .w_init

    % Load dataset (assumed to include x_train, y_train, x_test, y_test)
    M = load('ijcnn1.mat');

    % ---------------------
    % Preprocess Training Data
    % ---------------------
    D = M.x_train;
    [n, d] = size(D);

    % Normalize training features (zero mean, unit variance)
    s = std(D);
    s(s == 0) = 1;
    m = mean(D);
    D = (D - m) ./ s;

    % Add bias term
    D = [D, ones(n, 1)];

    % Shuffle training data
    rng(seed);
    perm = randperm(n);
    A = D(perm, :);
    B = M.y_train(perm);

    % Store transposed for column-wise access
    data.x_train = A';
    data.y_train = B';

    fprintf('This is Ijcnn1 train data with n=%d, d=%d\n', size(data.x_train'));

    % ---------------------
    % Preprocess Test Data
    % ---------------------
    P = M.x_test;
    [e, ~] = size(P);

    s = std(P);
    s(s == 0) = 1;
    m = mean(P);
    P = (P - m) ./ s;

    % Add bias term
    P = [P, ones(e, 1)];

    % Shuffle test data
    rng(seed);
    per = randperm(e);
    data.x_test = P(per, :)';
    data.y_test = M.y_test(per)';

    fprintf('This is Ijcnn1 test data with n=%d, d=%d\n', size(data.x_test'));

    % ---------------------
    % Initialize model weights
    % ---------------------
    rng(seed);
    data.w_init = randn(d + 1, 1);

end
