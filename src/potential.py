import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#fig=plt.figure()
#ax=Axes3D(fig)
X=np.arange(-4*np.pi,4*np.pi,0.1)
Y=np.arange(-4*np.pi,4*np.pi,0.1)

XX,YY=np.meshgrid(X,Y)

V1 = 2.0
V2 = 0.2
V=-V1-V1/2.0*(np.cos(XX)+np.cos(YY))+ \
    2.0*V2*(np.cos(0.5*XX)*np.cos(0.5*YY))

#ax.plot_surface(X,Y,V,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
#plt.contourf(X,Y,V,cmap='rainbow')
plt.pcolor(X,Y,V,cmap='d')
#plt.pcolormesh(X,Y,V,cmap='')
plt.axis('square')
plt.show()

