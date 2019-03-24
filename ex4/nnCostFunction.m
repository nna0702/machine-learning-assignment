function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Change format of output
out = zeros(m,num_labels);
for i = 1:m
    out(i, y(i)) = 1;
end

% Construct prediction from the model
pred = sigmoid([ones(m,1) sigmoid([ones(m,1) X] * Theta1')] * Theta2');

% Cost function
for i = 1:m
    J = J + 1/m * (- out(i,:) * log(pred(i,:)') - (1-out(i,:)) * log(1 - pred(i,:))');
end

% Initialise two components of the regularised term
reg1 = 0;
reg2 = 0;

% Remove the parameters of the bias units
Theta1_reg = Theta1(:,2:end);
Theta2_reg = Theta2(:,2:end);

% Sum of squares of parameters
for i = 1 : hidden_layer_size
    reg1 = reg1 + Theta1_reg(i,:) * Theta1_reg(i,:)'
end

for j = 1 : num_labels
    reg2 = reg2 + Theta2_reg(j,:) * Theta2_reg(j,:)'
end

% Regularised cost function
J = J + lambda/(2*m) * (reg1 + reg2);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% Initialise delta
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

% Compute gradients for each layer and example
for i = 1:m
    
    % Compute units
        X_temp = [ones(m,1) X];
        a1 = X_temp(i,:)';
        z2 = Theta1 * a1;
        a2 = [1; sigmoid(z2)];
        
    % Gradient of output layer
        delta3 = pred(i,:)' - out(i,:)';
    
    % Gradent of hidden layer
        delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
        delta2 = delta2(2:end);
       
    % Accumulate gradients
        D2 = D2 + delta3 * a2';
        D1 = D1 + delta2 * a1';
end




% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Gradients
Theta1_grad = 1/m * D1 + lambda/m * [zeros(size(Theta1,1),1) Theta1(:, 2:end)];
Theta2_grad = 1/m * D2 + lambda/m * [zeros(size(Theta2,1),1) Theta2(:, 2:end)];

% Add regularisation term
%Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m * Theta1_grad(:, 2:end);
%Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m * Theta2_grad(:, 2:end);














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
