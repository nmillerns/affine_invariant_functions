close all;
[X, Y] = meshgrid(-8:.025:8, -8:.025:8);
R = sqrt(X.^2+Y.^2)/10;
Angle = (atan2(Y,X) + pi)/2/pi;
FXY = sin(pi*R.*2.^(-Angle)./(2.^floor(log(R)/log(2)-Angle)) + pi);
surf(X, Y, FXY );
shading interp;
title('Snail Shell');
input('Hit Enter To Exit');
