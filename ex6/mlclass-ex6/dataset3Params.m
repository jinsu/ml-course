function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_tries = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_tries = [0.01 0.03 0.1 0.3 1 3 10 30];
err_mat = zeros(length(C_tries), length(sigma_tries));

%(C1 s1), (C1, s2) (C1, s3)...(C2, s1)
for i = 1:length(C_tries)
	for j = 1:length(sigma_tries)
		model = svmTrain(X, y, C_tries(i), @(x1, x2) gaussianKernel(x1, x2, sigma_tries(j)));
		prediction = svmPredict(model, Xval);
		err = mean(double(prediction ~= yval));
		err_mat(i, j) = err;
	end
end

err_mat
[row, col] = find(err_mat == min(min(err_mat)));
row
col
C = C_tries(row)
sigma = sigma_tries(col)

% =========================================================================

end
