import numpy as np 
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import root_scalar
from boundary_edges import alpha_shape

## plotting settings
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({"font.size":14})

###################################################################################################
#
# Setting experimental constants
#
###################################################################################################

## we consider five competition scenarios between the three types of Kaps of the form "X on Y background"
X_label = ['CRM1', 'Imp5', 'Imp5', 'KapB1', 'KapB1']
Y_label = ['KapB1', 'KapB1', 'CRM1', 'CRM1', 'Imp5']

## experimentally measured Delta values for the five competition scenarios
Delta_ex = [51, 84, 56, 37.5, 27.7]
## corresponding percentage error on Delta measurements
Delta_pm = [23, 34, 35, 16.7, 22.3] 

## If Kap X is smaller than Kap Y, the corresponding value of "flipped" is set to True
flipped = [False, False, True, True, True]
## ratio of volume of larger kap to volume of smaller kap
v = [1.3, 1.2, 1.1, 1.3, 1.2]


## constructing a lookup table for the min and max KD values consistent with experiments for the 3 types of Kaps
## columns are CRM1, Imp5, KapB1, the first row holds the min KD values, the second row holds the max
KD_lims = np.array([[0.118, 0.336], [0.1, 0.456], [0.1, 0.638]]) # min and max KD values for Kaps from SPR data
KD_lims = KD_lims.T
## column numbers of the KD lookup table for [X, Y] for each of the five competition experiments
kap_pairs = np.array([[0, 2], [1, 2], [1, 0], [2, 0], [2, 1]])

###################################################################################################
#
# Equations
#
###################################################################################################

## single kap accumulation curve
def phi(c, KD):
	return c/(c+KD)

## mixed kap accumulation curve (larger kap)
def phi1(x, v, c1, c2, KD1, KD2):
	return x*(1-x)**(v-1)/((1-x)*(1-c2/(c2+KD2)))**v - c1/KD1

## mixed kap accumulation curve (smaller kap)
def phi2(phi1, c2, KD2):
	return (1-phi1)*c2/(c2+KD2)

###################################################################################################
#
# Finding values of KDs consistent with all experimental competition outcomes
#
###################################################################################################

## 1 indicates smaller kap, 2 indicates larger kap
KD1 = np.linspace(0.1,1)
KD2 = np.linspace(0.1,1)
c1 = 10
c2 = 10
s_vf1 = c1/(c1+KD1)
s_vf2 = c2/(c2+KD2)

Delta = np.zeros([len(Delta_ex), len(KD1), len(KD2)])

## compute Delta for each competition scenario, for complete range of KD1 and KD2 
for e in range(len(Delta_ex)):

	for i in range(len(KD1)):
		for j in range(len(KD2)):

			## phi1 = solv.root
			solv = root_scalar(phi1, args=(v[e], c1, c2, KD1[i], KD2[j]), bracket=[0,0.999], x0=0.5)
	
			## phi2
			vf2 = phi2(solv.root, c2, KD2[j])

			if flipped[e]:
				## Delta of X, smaller particle
				Delta[e, i, j] = (s_vf2[j] - vf2)*100 # Y is type 1

			else:
				## Delta of X, bigger particle
				Delta[e, i, j] = (s_vf1[i] - solv.root)*100 # Y is type 2	


## this streak grows if the allowed KD limits do not change, indicating that 
## the permitted KD region has between the competition scenarios
no_change_streak = 0

## we require no changes for two iterations through all the comeptition scenarios to ensure convergence
while no_change_streak <= 2*len(Delta_ex):

	## cycle through experiments
	for e in range(len(Delta_ex)):

		## which KDs does this experiment involve?
		kp = kap_pairs[e]
		if flipped[e]:
			KD1_lims = KD_lims[:,kp[1]] # KD range of bigger Kap
			KD2_lims = KD_lims[:,kp[0]] # KD range of smaller Kap
		else:
			KD1_lims = KD_lims[:,kp[0]] # KD range of bigger Kap
			KD2_lims = KD_lims[:,kp[1]] # KD range of smaller Kap

		## placeholder min and max values
		min_KD1 = 1.0
		max_KD1 = 0.1
		min_KD2 = 1.0
		max_KD2 = 0.1

		## check if these limits are the placeholder max/min (i.e. no lims have been set)
		default = [True, True]
		if KD1_lims[0] < 1 or KD1_lims[1] > 0.1:
			default[0] = False
		if KD2_lims[0] < 1 or KD2_lims[1] > 0.1:
			default[1] = False

		for i in range(len(KD1)):
			for j in range(len(KD2)):	

				## if both KD lims have not been set, look at all cases of Delta within Delta_pm
				if default[0] and default[1]:
					if abs(Delta[e,i,j] - Delta_ex[e]) <= Delta_pm[e]:
						if KD1[i] < KD1_lims[0]:
							KD1_lims[0] = KD1[i]
						elif KD1[i] > KD1_lims[1]:
							KD1_lims[1] = KD1[i]
						if KD2[j] < KD2_lims[0]:
							KD2_lims[0] = KD2[j]
						elif KD2[j] > KD2_lims[1]:
							KD2_lims[1] = KD2[j]

				## if KD1 lims are not set, but KD2 lims are set, find biggest range of KD1 within allowed KD2 range
				elif default[0] and not default[1]:
					if KD2[j] >= KD2_lims[0] and KD2[j] <= KD2_lims[1]:
						if abs(Delta[e,i,j] - Delta_ex[e]) <= Delta_pm[e]:
							if KD1[i] < KD1_lims[0]:
								KD1_lims[0] = KD1[i]
							elif KD1[i] > KD1_lims[1]:
								KD1_lims[1] = KD1[i]

				## if KD1 lims are set, but KD2 lims are not set, find biggest range of KD2 within allowed KD1 range
				elif not default[0] and default[1]:
					if KD1[i] >= KD1_lims[0] and KD1[i] <= KD1_lims[1]:
						if abs(Delta[e,i,j] - Delta_ex[e]) <= Delta_pm[e]:
							if KD2[j] < KD2_lims[0]:
								KD2_lims[0] = KD2[j]
							elif KD2[j] > KD2_lims[1]:
								KD2_lims[1] = KD2[j]

				## if both KD1 and KD2 lims are set, calculate the range allowed for both
				## and compare to current ranges, taking the smallest range
				## if no consistent values exist, KD lims will reset to placeholder values

				else:

					if KD2[j] >= KD2_lims[0] and KD2[j] <= KD2_lims[1]:
						if abs(Delta[e,i,j] - Delta_ex[e]) <= Delta_pm[e]:
							if KD1[i] < min_KD1:
								min_KD1 = KD1[i]
							elif KD1[i] > max_KD1:
								max_KD1 = KD1[i]

					if KD1[i] >= KD1_lims[0] and KD1[i] <= KD1_lims[1]:
						if abs(Delta[e,i,j] - Delta_ex[e]) <= Delta_pm[e]:
							if KD2[j] < min_KD2:
								min_KD2 = KD2[j]
							elif KD2[j] > max_KD2:
								max_KD2 = KD2[j]

		## if both KD1 and KD2 lims are set, compare to current ranges and take smallest range
		if not default[0] and not default[1]:
			KD1_lims[0] = max(KD1_lims[0], min_KD1)
			KD1_lims[1] = min(KD1_lims[1], max_KD1)
			KD2_lims[0] = max(KD2_lims[0], min_KD2)
			KD2_lims[1] = min(KD2_lims[1], max_KD2)

		prev_KD_lims = KD_lims # old limits, to be compared with new limits for deciding convergence

		## update new KD lims
		if flipped[e]:
			KD_lims[:,kp[1]] = KD1_lims
			KD_lims[:,kp[0]] = KD2_lims
		else:
			KD_lims[:,kp[0]] = KD1_lims
			KD_lims[:,kp[1]] = KD2_lims

		## if comparison to this competition scenario has not changed KD lims, grow streak by 1
		## otherwise reset streak
		if np.equal(prev_KD_lims, KD_lims).all():
			no_change_streak += 1
		else:
			no_change_streak = 0


###################################################################################################
#
# Plotting Delta for all five competition scenarios as functions of the KD values
# Plotting equi-Delta contours and KD regions permitted by experiments
#
###################################################################################################

for e in range(len(Delta_ex)):

	title = "{0} with {1} background".format(X_label[e], Y_label[e])
	boundary = [] ## collection of all points in KD space which are within permitted KD range

	## which KD boundaries to use?
	kp = kap_pairs[e]
	if flipped[e]:
		KD1_lims = KD_lims[:,kp[1]] # bigger particle
		KD2_lims = KD_lims[:,kp[0]] # smaller particle
	else:
		KD1_lims = KD_lims[:,kp[0]] # bigger particle
		KD2_lims = KD_lims[:,kp[1]] # smaller paritcle

	for i in range(len(KD1)):
		for j in range(len(KD2)):

			## finding the collection of all points in KD space within the permitted range
			if flipped[e]:
				if KD1[i] > KD1_lims[0] and KD1[i] < KD1_lims[1] and KD2[j] > KD2_lims[0] and KD2[j] < KD2_lims[1]:
					if abs(Delta[e,i,j] - Delta_ex[e]) <= Delta_pm[e]:
						boundary.append([KD1[i],KD2[j]])

			else:
				if KD1[i] > KD1_lims[0] and KD1[i] < KD1_lims[1] and KD2[j] > KD2_lims[0] and KD2[j] < KD2_lims[1]:
					if abs(Delta[e,i,j] - Delta_ex[e]) <= Delta_pm[e]:
						boundary.append([KD2[j], KD1[i]])


	## computing the bounding edge of all points within permitted KD range
	boundary = np.array(boundary)
	edges = alpha_shape(boundary, alpha=1, only_outer=True)

	## contour levels which are experimental Delta values, and +/- error
	levels = [Delta_ex[e]-Delta_pm[e], Delta_ex[e], Delta_ex[e]+Delta_pm[e]]

	ax = plt.subplot()

	## plotting Delta heatmaps and contours
	if flipped[e]:
		im = ax.imshow(Delta[e].T, extent=[min(KD2),max(KD2),max(KD1),min(KD1)], aspect="auto")
		ct = ax.contour(KD2, KD1, Delta[e].T, levels, colors=['grey', 'white', 'grey'], linestyles='dashed')
	else:
		im = ax.imshow(Delta[e], extent=[min(KD2),max(KD2),max(KD1),min(KD1)], aspect="auto") 
		ct = ax.contour(KD2, KD1, Delta[e], levels, colors=['grey', 'white', 'grey'], linestyles='dashed')
	ax.set_xlabel(Y_label[e] + " $K_D [\mu M]$")
	ax.set_ylabel(X_label[e] + " $K_D [\mu M]$")
	ax.set_title(title)
	plt.clabel(ct, fmt='%d')

	## plotting permitted KD region
	for i, j in edges:
		ax.plot(boundary[[i, j], 0], boundary[[i, j], 1], c='k', linewidth=2)

	## colourbar
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	cax.set_title("$\Delta$")

	plt.colorbar(im, cax=cax)
	plt.tight_layout()
	plt.show()
