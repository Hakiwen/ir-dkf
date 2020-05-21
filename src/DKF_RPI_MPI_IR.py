# mpiexec -np 9 xterm -e python3 DKF_RPI_MPI.py
# mpiexec -n 9 python3 DKF_RPI_MPI.py
#mport mpi4py
#pi4py.rc.recv_mprobe = False
from mpi4py import MPI

import numpy as np
import time
from datetime import datetime

import Adafruit_GPIO.SPI as SPI
import Adafruit_MCP3008

#import matplotlib.pyplot as plt

np.set_printoptions(precision=2)

CLK  = 18
MISO = 23
MOSI = 24
CS   = 25
mcp  = Adafruit_MCP3008.MCP3008(clk=CLK, cs=CS, miso=MISO, mosi=MOSI)

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
print('My rank is ', rank)
print('My size is ', size)

#model for sensor readings
#f(x) = a*exp(b*x) + c*exp(d*x)
a = 3175
b = -0.01686
c = 91.96
d = -0.002468


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
		self.x_bar = np.matrix(str(20*(np.random.random_sample() - 0.5)) + ';' + str(20*(np.random.random_sample() - 0.5))) + self.P*np.random.randn(2, 1)
		self.message_out = {'u':1/self.R * H.transpose() * self.z, 'U':self.U, 'x_bar':self.x_bar}
		self.message_in = [{'u':self.message_out['u'], 'U':self.message_out['U'], 'x_bar':self.message_out['x_bar']} for i in range(len(E))]
		print(self.x_bar)
def sensorsCircConfig(n, ind):
	# xs: sensors' positions in global ref
	# H: observation matrix
	# Adj: Adjacency vector
	# E: edges
    # th: sensors' orientations in global ref

	sensors_array_radius = 40 # sensor-to-center distance
	phi = np.linspace(2*np.pi - 2*np.pi/n, 0.0, num = n)
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
#	print(str(rank) + ',' + str(x_hat))	
	
	s.P = s.A*M*s.A + s.B*s.Q*s.B.transpose() # covariance update
	s.x_bar = s.A*x_hat # corection
	
#	print(str(rank) + ',' + str(s.x_bar))	
#	time.sleep(5)
	s.message_out['u'] = s.u
	s.message_out['U'] = s.U
	s.message_out['x_bar'] = s.x_bar

	return s


def convert_to_distance(v):
	z = a*np.exp(v*b) + c*np.exp(v*d)
	return z;

def check_reading(s,z):
	if z >= 15 and z <= 60:
		return z;
	else:
		return s.H*s.x_bar

def sensor_measure(s, dist): # must be an actual sensor reading when using hw
#	w = s.R*np.random.randn(2, 1) # Sample noise

	# convert real measure into sensor distance measurment
#	x_local = xtrue[0,0] - s.xs[0,0]
#	y_local = xtrue[1,0] - s.xs[1,0]

	# verify if target is intersecting sensor's beam
#	p1 = s.xs;
#	p2 = s.xs + 900.0*np.matrix([np.cos(s.th), np.sin(s.th)]).transpose()
#	dist = np.absolute( ( p2[1]-p1[1] )*xtrue[0,0] - ( p2[0]-p1[0] )*xtrue[1,0] + p2[0]*p1[1] - p2[1]*p1[0] ) / np.sqrt( ( p2[1]-p1[1])**2 + ( p2[0]-p1[0] )**2 );
	
	if dist <= 60:                                         # if robots is within range
#		ys = np.sqrt(x_local**2 + y_local**2)                 # actual range measure on sensor
		x_global = s.xs + dist*np.matrix([np.cos(s.th), np.sin(s.th)]).transpose()    # what sensor thinks is global position of target
	else:
		x_global = s.x_bar

	z = s.H*x_global
	return z

################################################################################################################################

""" Mains """

if rank == 0: # node0 logs results	
#	import matplotlib.pyplot as plt
	f_e = open(datetime.now().strftime('e_%H_%M_%S_%d_%m_%Y.log'),'w')
	f_m = open(datetime.now().strftime('m_%H_%M_%S_%d_%m_%Y.log'),'w')
#	fig = plt.figure()
#	ax = fig.add_subplot(111, autoscale_on=False, xlim=(-100, 100), ylim=(-100,100))
#	ax.grid()
#	agPlt, = ax.plot([],[],'o')

	while 1:
#		sim_data = x
#		for i in range(1,size):
#			req = comm.isend(sim_data, dest=i)
#			req.wait()
		#print(x, rank)

#		w = Q*np.random.randn(2, 1)
#		x = A*x + B*w
		recv_log_e = ['']*(size)
		recv_log_m = ['']*(size)
		recv_logt = ['']*(size)
		str_e = str(time.time()) + '|'
		str_m = str_e
		

		for i in range(1,size):
#			recv_logt[i] = comm.irecv(source = i)
#			recv_valt = recv_logt[i].wait()	
			recv_log_e[i] = comm.irecv(source = i)
			recv_val_e = recv_log_e[i].wait()
			recv_log_m[i] = comm.irecv(source = i)
			recv_val_m = recv_log_m[i].wait()
			str_e = str_e  + "%5.2f" % recv_val_e[0] + ',' + "%5.2f" % recv_val_e[1] + '|'
			str_m = str_m  + "%10d" %recv_val_m + ','
#		agPlt.set_xdata(recv_val_e[:,0])
#		agPlt.set_ydata(recv_val_e[:,1])
#		plt.draw()
		str_e = str_e
		str_m = str_m
		f_e.write(str_e + "\n")
		f_m.write(str_m + "\n")
		print(str_e)
#		print(str_m)
#		time.sleep(dt)		
else:
	sim_data = None
	n = size - 1 # excluding node0
	xs, H, Adj, E, th = sensorsCircConfig(n, rank - 1)
	sensor = Sensor_Init(rank-1, H, dt, A, B, Q, n, xs[:,rank-1], th[rank-1], E)
	n_recv = ['']*len(E)
	n_send = ['']*len(E)
	while 1:    	
#		req = comm.irecv(source=0)
#		sim_data = req.wait()
		
		#sensor.z = sensor_measure(sensor, sim_data) # must be changed to an acctual infrared sensor	
		sensor_sample = mcp.read_adc(0)
		distance = convert_to_distance(sensor_sample)
		sensor.z = sensor_measure(sensor,distance)

		for i in range(len(E)):
			n_send[i] = comm.isend(sensor.message_out, dest=E[i,0]+1)
			n_send[i].wait()

		for i in range(len(E)):
			n_recv[i] = comm.irecv(source=E[i,0]+1)
			sensor.message_in[i] = n_recv[i].wait()

		sensor = filter_update(sensor)
#		timestamp = time.time()
#		send_hostt = comm.isend(str(timestamp), dest=0)	        
		send_host_e = comm.isend(sensor.x_bar, dest=0)
		send_host_m = comm.isend(distance, dest=0)
		#err = np.sqrt((sim_data[0] - sensor.x_bar[0])**2 + (sim_data[1] - sensor.x_bar[1])**2) / np.sqrt((sim_data[0])**2 + (sim_data[1])**2) * 100
		#print('percent err: ', err, ', rank: ', rank)		
#		print('sensor reading: ', convert_to_distance(sensor_sample), ', rank: ', rank)
		time.sleep(dt)

while 1:
	# does nothing but smiling at you :)
	pass
