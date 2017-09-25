import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

P = 100.0*np.eye(9)

#define dt = 0.05 rather than dt = 1
dt = 0.05
A = np.matrix([[1.0, 0.0, 0.0, dt, 0.0, 0.0, 1/2.0*dt**2, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

#define measurement/observation matrix
H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

#define observation noise covariance
R = 100*np.eye(3)

#define process noise matrix Q
#from the blog we mentioned how to compute Q from G
G = np.matrix([[0.5*dt**2],[0.5*dt**2],[0.5*dt**2],[dt],[dt],[dt],[1],[1],[1]])
a = 0.1
Q = G*G.T*a**2
print(Q)
print(Q.shape)

#system initialization
px=0.0
py=0.0
pz=0.0

vx=1
vy=2
vz=3

Xr=[]
Yr=[]
Zr=[]

# #generate position:
for i in range(100):

    # we assume constant acceleratoin for this demo
    accx = 3
    vx += accx * dt
    px += vx * dt

    accy = 1
    vy += accy * dt
    py += vy * dt

    accz = 1
    vz += accz * dt
    pz += vz * dt

    Xr.append(px)
    Yr.append(py)
    Zr.append(pz)

    # position noise
    sp = 0.5

Xm = Xr + sp * (np.random.randn(100))
Ym = Yr + sp * (np.random.randn(100))
Zm = Zr + sp * (np.random.randn(100))


# fig = plt.figure(figsize=(16,9))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(Xm, Ym, Zm, c='gray')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.title('Observed/Measured Trajectory (with Noise)')
#
# # Axis equal
# max_range = np.array([Xm.max()-Xm.min(), Ym.max()-Ym.min(), Zm.max()-Zm.min()]).max() / 3.0
# mean_x = Xm.mean()
# mean_y = Ym.mean()
# mean_z = Zm.mean()
# ax.set_xlim(mean_x - max_range, mean_x + max_range)
# ax.set_ylim(mean_y - max_range, mean_y + max_range)
# ax.set_zlim(mean_z - max_range, mean_z + max_range)
# plt.show()

#stack measurements
measurements = np.vstack((Xm,Ym,Zm))
print(measurements.shape)

#define initial state
X = np.matrix([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 1.0, 1.0]).T
I = np.eye(9)
Xx = []
Xy = []
Xz = []

for each_step in range(100):

    #prediction step:
    X = A*X
    #state covariance propogation
    P = A*P*A.T + Q

    #measurement covariance propogation
    S = H*P*H.T + R

    #Kalman Gain update
    #pseudo-inverse here
    K = (P*H.T) * np.linalg.pinv(S)
    # print(K)
    Z = measurements[:,each_step].reshape(3,1)

    #corrected/updated x
    X = X + K*(Z-(H*X))

    #update state covariance P:
    P = (I - (K * H)) * P

    #store corrected/updated x
    Xx.append(float(X[0]))
    Xy.append(float(X[1]))
    Xz.append(float(X[2]))




fig = plt.figure(figsize=(16,9))

plt.plot(Xx,Xz, label='Filtered result')
plt.plot(Xr, Zr, label='Actual Trajectory')
plt.scatter(Xm,Zm, label='Measurement/Observed trajectory', c='gray', s=30)
plt.title('Trajectory Comparison between Actual,Measured and KF')

plt.legend(loc='best',prop={'size':22})
plt.axhline(0, color='k')
plt.axis('equal')
plt.xlabel('X ($m$)')
plt.ylabel('Y ($m$)')
plt.ylim(0, 15);
plt.show()
