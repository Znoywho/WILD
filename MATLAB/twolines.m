x = linspace(0,2*pi);
y = sin(x);

plot(x,y,"b")
% use `hold on` meaning stopping and waiting until `hold off` trigger
hold on

y2 = cos(x)
plot(x,y2,":" )
legend("sin", "cos")

hold off

