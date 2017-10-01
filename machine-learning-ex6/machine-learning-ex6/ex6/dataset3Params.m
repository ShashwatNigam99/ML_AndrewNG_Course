function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C_list = [0.001 0.01 0.1 1 10 100];
sigma_list = [0.001 0.01 0.1 1 10 100];
ans = [1,C,sigma];

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

for i = 1:size(C_list,2)
	for j = 1 :size(sigma_list,2)
		model= svmTrain(X, y, C_list(i), @(x1, x2) gaussianKernel(x1, x2, sigma_list(j)));
		pred = svmPredict(model,Xval);
		predictions = mean(double(pred ~= yval)); 
		if (predictions < ans(1))
			ans(1) = predictions;
			ans(2) = C_list(i);
			ans(3) = sigma_list(j);
		end	
	end
end

C = ans(2);
sigma = ans(3);



% =========================================================================

end
