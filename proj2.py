import math
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy as np
from matplotlib import rc

v_zero = 1.0 #Free flow velocity
nu = .1 #Viscosity
a = 1.0
L= 64
h = a / (L+1)

def set_grid():
	A = np.zeros((L+1,L+1))
	w = np.zeros((L+1,L+1))

	x_list = []
	y_list = []
	
	for j in range(L+1): #rows
		x_list.append(float(j)/float(L))
		y_list.append(float(j)/float(L))
		for l in range(L+1): #columns
			y = (L - j) * L**-1
			if ((j==L) and ((l<0.25*L) or (l>0.375*L))) or ((j==0) and ((l<0.65*L) or (l>0.8*L))): #Boundaries A & E
				A[j][l] = 0.0
				w[j][l] = 0.0	
			elif ((0.25*L) <= l <= (0.375*L) and (j>=0.75*L)) or ((0.65*L) <= l <= (0.8*L) and (j<=.3*L)): #The plate
				A[j][l] = 0.0
				w[j][l] = 0.0
			else:
				A[j][l] = v_zero * y
				w[j][l] = 0.0
	print()
	print()
	print()
	print(x_list)
	print(y_list)
	print()
	print()
	print()
	return A,w,x_list,y_list

one, two, x_list, y_list = set_grid()
print(one)
print(two)

def sweep(A,w,relax_par=1.5):

	residual_array = []
	norm = 0.0
	#Updating the interior. The reason why the arange is from 1 to L and not L-1 is 
	#because the downstream boundary condition is also edited in this loop to make things easier. 
	#relax_par is the w value from the class notes, changed so that it isn't confused with omega 
	#which is referred to as w.
	for j in np.arange(1,L-1): #rows
		for l in np.arange(0,L): #columns
			if A[j][l] != 0:	
				A[j][l] = (A[j][l] * (1-relax_par) + (0.25*relax_par) * ((A[j+1][l] + A[j-1][l] + A[j][l-1] + A[j][l+1]) + (w[j][l])*h**2))
			#Updating the boundaries. Stream function is always 0 at boundaries A, E, B, C, D. It is always v_0*y at boundaries F and H. 
			elif (l==0): #Boundary F
				A[j][l] = v_zero * y
				w[j][l] = 0.0
			elif (l==L): #Boundary H
				A[j][l] = A[j][l-1]


	for j in np.arange(1,L): #rows
		for l in np.arange(1,L): #columns
			y = (L - j) * L**-1
			if ((l==(0.25*L)) and (j>=(0.75*L))) or ((l==(0.65*L)) and (j<=(0.3*L))): #Boundary D
				w[j][l] = -2. * A[j][l-1] / h**2
				#print(w[j][l], 'left', (j,l) )
			elif ((l==(3.*L/8.)) and (j>=(3.*L/4.))) or ((l==(.8*L)) and (j<=(0.3*L))): #Boundary B
				w[j][l] = -2. * A[j][l+1] / h**2
				#print(w[j][l], 'right', (j,l) )
			elif ((L/4.) < l < (3.*L/8.) and (j==(3.*L/4.))) or ((0.65*L) < l < (0.8*L) and (j==(0.3*L))): #Boundary C
				w[j][l] = -2. * A[j-1][l] / h**2
				#print( w[j][l], 'up', (j,l))
			elif (l<=(L/4.) or l>=(3.*L/8.)) and j<=(3.*L/4.):
				psi_partial_x = (A[j][l+1] - A[j][l-1]) / (2*h)
				psi_partial_y = (A[j+1][l] - A[j-1][l]) / (2*h)
				omega_partial_x = (w[j][l+1] - w[j][l-1]) / (2*h)
				omega_partial_y = (w[j+1][l] - w[j-1][l]) / (2*h)
				w[j][l] = (w[j][l] * (1-relax_par)) + (0.25*relax_par) * ((w[j+1][l] + w[j-1][l] + w[j][l-1] + w[j][l+1]) - ((nu**-1) * ((psi_partial_y * omega_partial_x) - (psi_partial_x * omega_partial_y)))*h**2)
				#print w[j][l], (j,l)
			elif (l==0): #Boundary F
				w[j][l] = 0.0
			elif (j==0): #Boundary G
				w[j][l] = 0.0
			elif (l==L): #Boundary H
				w[j][l] = w[j][l-1]
			residual = h**2 * A[j][l] + w[j][l]
			residual_array.append(residual)
			norm += residual**2
	norm = norm**(1.0/2.0)
	return A,w, residual_array, norm
sweepa, sweepw, r, n = sweep(one,two)
#print sweepa
#print sweepw
sweepa2, sweepw2, r2, n2 = sweep(sweepa,sweepw)
#print sweepa2
#print sweepw2

def sweeps(A,w,n):
	i=0
	residual_array = []
	norm_array = []
	while i<n:
		A, w, residual, norm = sweep(A,w)
		i+=1
	return A,w,residual,norm


def graphA(x_list, y_list, matrix):
	# use latex...
	# rc('text', usetex = True)
	# font = {'family' : 'normal','weight' : 'normal', 'size' : 12 }
	# plt.rc('font', **font)

	plt.figure()
	PLOT = plt.contour(x_list, y_list, matrix, 12)
	plt.clabel(PLOT, inline=1, fontsize=10)
	plt.xlabel(r'$x$')
	plt.ylabel(r'$y$')
	plt.title(r'Contour Plot of Stream Function $\psi$')
	plt.show()

def graphW(x_list, y_list, matrix):
	# use latex...
	# rc('text', usetex = True)
	# font = {'family' : 'normal','weight' : 'normal','size' : 12 }
	# plt.rc('font', **font)

	plt.figure()
	PLOT = plt.contour(x_list, y_list, matrix, 12)
	plt.clabel(PLOT, inline=1, fontsize=10)
	plt.xlabel(r'$x$')
	plt.ylabel(r'$y$')
	plt.title(r'Contour Plot of vorticity $\omega$')
	plt.show()



swepta, sweptw, sweptr, sweptn = sweeps(one, two, 80)
print(swepta)
print(sweptw)
graphA(x_list, y_list, swepta)
graphW(x_list, y_list, sweptw)




