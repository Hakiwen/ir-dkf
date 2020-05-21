# mpiexec -np 9 xterm -e python3 DKF_RPI_MPI.py
# mpiexec -n 9 python3 DKF_RPI_MPI.py

from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
print('My rank is ', rank)
print('My size is ', size)

dt = 0.0015
A0 = 4*np.matrix('0, -1; 1, 0')       # Task's continuous time dynamics
B0 = 25*np.matrix('1 0; 0 1')

# Discretization
A = np.matrix('1 0; 0 1') + dt*A0 + (dt**2/2)*A0*A0 + (dt**3/6)*A0*A0*A0
B = dt*B0
Q = 15*np.matrix('1 0; 0 1')
x = np.matrix('140; 0')

################################################################################################################################

class Sensor_Init(object):
	def __init__(self, ind, H, dt, A, B, Q, n, xs, th, E):
		super(Sensor_Init, self).__init__()
		self.xs = xs
		self.P = np.matrix('1 0; 0 1')
		self.z = np.matrix('0; 0')
		self.th = th
		self.u = np.matrix('0; 0')
		self.dt = dt
		self.A = A
		self.B = B
		self.Q = Q
		self.H = H
		self.R = 20*np.sqrt(ind+1)
		self.U = 1/self.R * H.transpose() * H
		self.x_bar = np.matrix('15; -10') + self.P*np.random.randn(2, 1)
		self.message_out = {'u':1/self.R * H.transpose() * self.z, 'U':self.U, 'x_bar':self.x_bar}
		self.message_in = [{'u':self.message_out['u'], 'U':self.message_out['U'], 'x_bar':self.message_out['x_bar']} for i in range(len(E))]

def sensorsCircConfig(n, ind):
	# xs: sensors' positions in global ref
	# H: observation matrix
	# Adj: Adjacency vector
	# E: edges
    # th: sensors' orientations in global ref

	sensors_array_radius = 400 # sensor-to-center distance
	phi = np.linspace(0.0, 2*np.pi - 2*np.pi/n, num = n)
	alpha_offset = 20 # sensor offset angle (to avoid sensors interferying with each other)
	th = phi + np.pi + np.deg2rad(alpha_offset)

	xs = sensors_array_radius*np.matrix([np.cos(phi), np.sin(phi)])

	Adj = np.zeros((n,1))
	if ind == 0:
		Adj[[1,2,6,7]] = 1
	elif ind == 1:
		Adj[[0,2,3,7]] = 1
	elif ind == 5:
		Adj[[3,4,6,7]] = 1
	elif ind == 6:
		Adj[[4,5,7,0]] = 1
	else:
		ngb = np.mod((ind+1)+np.matrix('-2,-1,1,2'),n)-np.matrix('1,1,1,1')
		Adj[ngb] = 1

	E = np.argwhere(Adj)
	H = np.matrix('1 0; 0 1')

	return xs, H, Adj, E, th

def filter_update(s):
	m = len(s.message_in) # number of neighbots
	s.u = 1/s.R * s.H.transpose() * s.z # filtered measurement
	y = s.u
	S = s.U
	w_sum = np.matrix('0; 0')
	for i in range(m):
		y = y + s.message_in[i]['u']
		S = S + s.message_in[i]['U']
		w_sum = w_sum + (s.message_in[i]['x_bar'] - s.x_bar)

	M = np.linalg.inv(np.linalg.inv(s.P) + S)
	x_hat = s.x_bar + M*(y - S*s.x_bar) + s.dt*M*w_sum # prediction
	s.P = s.A*M*s.A + s.B*s.Q*s.B.transpose() # covariance update
	s.x_bar = s.A*x_hat # corection

	s.message_out['u'] = s.u
	s.message_out['U'] = s.U
	s.message_out['x_bar'] = s.x_bar

	return s

def sensor_measure(s, xtrue): # must be an actual sensor reading when using hw
	w = s.R*np.random.randn(2, 1) # Sample noise

	# convert real measure into sensor distance measurment
	x_local = xtrue[0,0] - s.xs[0,0]
	y_local = xtrue[1,0] - s.xs[1,0]

	# verify if target is intersecting sensor's beam
	p1 = s.xs;
	p2 = s.xs + 900.0*np.matrix([np.cos(s.th), np.sin(s.th)]).transpose()
	dist = np.absolute( ( p2[1]-p1[1] )*xtrue[0,0] - ( p2[0]-p1[0] )*xtrue[1,0] + p2[0]*p1[1] - p2[1]*p1[0] ) / np.sqrt( ( p2[1]-p1[1])**2 + ( p2[0]-p1[0] )**2 );
	
	if dist <= 25:                                         # if robots is within range
		ys = np.sqrt(x_local**2 + y_local**2)                 # actual range measure on sensor
		x_global = s.xs + ys*np.matrix([np.cos(s.th), np.sin(s.th)]).transpose()    # what sensor thinks is global position of target
	else:
		x_global = s.x_bar

	z = s.H*x_global + w
	print(z)
	return z

################################################################################################################################

""" Mains """

if rank == 0: # node0 simulate dynamics (not needed for a real hw)	
	while 1:
		sim_data = x
		for i in range(1,size):
			req = comm.isend(sim_data, dest=i)
			req.wait()
		print(x, rank)

		w = Q*np.random.randn(2, 1)
		x = A*x + B*w
		time.sleep(dt)		
else:
	sim_data = None
	n = size - 1 # excluding node0
	xs, H, Adj, E, th = sensorsCircConfig(n, rank - 1)
	sensor = Sensor_Init(rank-1, H, dt, A, B, Q, n, xs[:,rank-1], th[rank-1], E)
	n_recv = ['']*len(E)
	n_send = ['']*len(E)
	while 1:    	
		req = comm.irecv(source=0)
		sim_data = req.wait()
		
		sensor.z = sensor_measure(sensor, sim_data) # must be changed to an acctual infrared sensor	

		for i in range(len(E)):
			n_send[i] = comm.isend(sensor.message_out, dest=E[i,0]+1)
			n_send[i].wait()

		for i in range(len(E)):
			n_recv[i] = comm.irecv(source=E[i,0]+1)
			sensor.message_in[i] = n_recv[i].wait()

		sensor = filter_update(sensor)
		print(str(rank) + ',' + str(sensor.x_bar))
		err = np.sqrt((sim_data[0] - sensor.x_bar[0])**2 + (sim_data[1] - sensor.x_bar[1])**2) / np.sqrt((sim_data[0])**2 + (sim_data[1])**2) * 100
		#print('percent err: ', err, ', rank: ', rank)		
		time.sleep(dt)

while 1:
	# does nothing but smiling at you :)
	pass
