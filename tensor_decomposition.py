import pickle
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker, parafac, non_negative_tucker
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = pickle.load(open('data_mat3D.pickle','rb'), encoding='bytes')    
data = np.nan_to_num(data)
data.shape
# (195, 8, 12)

##### plot data in 3d
from mpl_toolkits.mplot3d import axes3d, Axes3D
fig = plt.figure()
ax = Axes3D(fig)
x_pos = np.empty(data.size)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
ax.scatter()


#### neuron over time for trial 1
plt.title('Neuron over Time for trial 1')
plt.xlabel('time slot')
plt.ylabel('spikes in each slot')
for i in range(195):
    plt.plot(data[i,1,:], alpha=0.3)
plt.show()

amean  = np.mean(data,axis=(1,2))
for i in range(195):
    data[i,:,:] -= amean[i]
    

#### mean removed
from mpl_toolkits.mplot3d import axes3d, Axes3D
fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(8)
y = np.arange(12)
X, Y = np.meshgrid(y, x)
ax.plot_wireframe(X, Y, data[10,:,:])
plt.title('mean removed')
plt.xlabel('trials')
plt.ylabel('time')
plt.show()


##### test on single trial
single = data[:,0,:]
core, factors = non_negative_tucker(single, [12, 12])
reconstructed = np.dot(np.dot(factors[0], core), np.transpose(factors[1]))
np.sum((reconstructed - single)**2)/single.size

plt.imshow(factors[0])

##### tuck decompostion on 3d data
core, factors = tucker(data, [8,8,8])
plt.subplot(131)
plt.imshow(factors[0])
plt.colorbar()
plt.subplot(132)
plt.imshow(factors[1])
plt.colorbar()
plt.subplot(133)
plt.imshow(factors[2])
plt.colorbar()
plt.show()

core.shape
x_coord = np.empty(core.size)
y_coord = np.empty(core.size)
z_coord = np.empty(core.size)
p_size = np.empty(core.size)
for i in range(core.shape[0]):
    for j in range(core.shape[1]):
        for k in range(core.shape[2]):
            x_coord[i*core.shape[1]*core.shape[2] + j*core.shape[2] + k] = i
            y_coord[i*core.shape[1]*core.shape[2] + j*core.shape[2] + k] = j
            z_coord[i*core.shape[1]*core.shape[2] + j*core.shape[2] + k] = k
            p_size[i*core.shape[1]*core.shape[2] + j*core.shape[2] + k] = core[i,j,k]* 5
fig = plt.figure()
ax = Axes3D(fig)
plt.xlabel('neurons')
plt.ylabel('stimulus category')
plt.zlabel('time slot')
ax.scatter(x_coord, y_coord, z_coord, s=np.abs(p_size))
plt.show()


time_good = np.zeros(8)
for i in range(8):
    time_good[i] = np.sum(core[:,:,:(i+1)]) / np.sum(core)
plt.title('Representation power on Time')
plt.xlabel('# of PCs')
plt.ylabel('Accum % of total variance')
plt.plot(time_good)
plt.show()

stimulus_good = np.zeros(8)
for i in range(8):
    stimulus_good[i] = np.sum(core[:,:(i+1),:]) / np.sum(core)
plt.title('Representation power on Stimulus')
plt.xlabel('# of PCs')
plt.ylabel('Accum % of total variance')
plt.plot(stimulus_good)
plt.show()


neurons_good = np.zeros(8)
for i in range(8):
    neurons_good[i] = np.sum(core[:(i+1),:,:]) / np.sum(core)
plt.title('Representation power on neurons')
plt.xlabel('# of PCs')
plt.ylabel('Accum % of total variance')
plt.plot(neurons_good)
plt.show()

            
####
core, factors = non_negative_tucker(data, [8, 8, 8])
p1 = np.dot(data, factors[2][:,:3])
p2 = np.rollaxis(p1, 1, start=3)
