x = linspace(-2, 2, 20);
y = x'; % `'` meaning is transpose of matrix
z = x .* exp(-x.^2 - y.^2);
% create the a surface plot
surf(x,y,z)

