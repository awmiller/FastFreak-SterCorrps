function surf_n_terp(xX,yY,zZ)
% SURF_N_TERP creats a surface from interpolated non-uniform samples,
% assuming zZ can be fit to some function of xX and yY
    f = scatteredInterpolant(xX,yY,zZ);
    X=linspace(min(xX),max(xX),100);
    Y=linspace(min(yY),max(yY),100);
    [X,Y] = meshgrid(X,Y);
    Z = f(X,Y);
    surf(X,Y,Z);
end