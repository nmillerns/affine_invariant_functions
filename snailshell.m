% Creates a "snail shell" like surface
% The surface is invariant to scale and rotation
% f(2^(dr/2/pi)*R(dTheta)X + 0) = f(X) for all dr > 0, dTheta > 0
% Where R(dTheta) is the rotation matrix rotating by angle dTheta
% Ie the resulting plot stays the same as it rotates and shrinks like a snail shell

close all; % close all windows for plotting
[X, Y] = meshgrid(-8:.025:8, -8:.025:8); % make a 2D domain [-8,8]x[-8,8]
R = sqrt(X.^2+Y.^2)/10; % Compute radius'
Angle = (atan2(Y,X) + pi)/2/pi; % Compute unit angles [0, 1]
FXY = sin(pi*R.*2.^(-Angle)./(2.^floor(log(R)/log(2)-Angle)) + pi); % Surface f(x,y)
surf(X, Y, FXY );
shading interp;
title('Snail Shell');
input('Hit Enter To Exit');
