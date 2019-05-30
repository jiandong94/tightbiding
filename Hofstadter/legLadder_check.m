
phi = pi/2.;
tprime = 1.0;
im = sqrt(-1);
I = [1 0;0 1];
sigmax = [0 1;1. 0];
sigmay = [0 -im;im 0];
sigmaz = [1 0;0 -1];
band = 2;
ndots = 201;
eigv = zeros(band,ndots);
kloop = linspace(-pi,pi,ndots);
H = zeros(2,2);
for i=1:ndots
    k = kloop(i);
    epsilon0 = cos(k)*cos(phi);
    xi = tprime/2;
    epsilon1 = sin(k)*sin(phi);
    H = -2*(epsilon0*I+xi*sigmax+epsilon1*sigmaz);
    eigv(:,i) = eigs(H);
end
plot(kloop,eigv(1,:),'k.')
hold on
plot(kloop,eigv(2,:),'k.')