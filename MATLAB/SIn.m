% setup of coordination of each plot
x = linspace(0,2*pi)
y = sin(x)



% plot on screen normally
% plot(x,y)

% Label the axes
xlabel("x")
ylabel("sin(x)")

% Title of graph
title("Plot of Sine Function")

plot(x,y, "b--")
%{
    b: meaaning color of line is blue
    --: dashed line
%}