#Plot optimal values, compare with data, plot it, and make a table.
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from lmfit import minimize, Parameters, report_fit
import pickle
import time
import glob
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rc('font', size=16)
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['mathtext.sf'] = 'Arial'

from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy import special

#New ODE, that accounts for product inhibition at two different steps.
#x is the d-luciferin concentratin
#y is luciferyl-adenylate 
#z is oxyluciferin
#E is the enzyme
#I'm assuming that light emission happens in the enzyme catalyzed step when yE is converted to zE
#Same as above, but now I'm inputting explicit Kd from the literature to maybe make this fit not as bananas to deal with.
#Kd is koff/kon, so all the "koff" terms can be replaced by Kd * kon
#kcat is related to Km as well. Km = koff + kcat/kon

def kdode(wvec, t, pvec):
	#x, xE, y, yE, z, zE, E = wvec # current values for the concentrations
	#konx, kony, konz, kdx, kdy, kdz, kmx, kmy = pvec
	x, xE, y, yE, E = wvec # current values for the concentrations
	konx, kony, kdx, kdy, kmx, kmy = pvec
	f = [konx * (kdx * xE - x * E),
		konx * (x * E - kmx * xE),
		kony * (kdy * yE - y * E),
		kony * (y * E - kmy * yE) + konx * (kmx - kdx) * xE,
		#konz * (kdz * zE - z * E),
		#konz * (z * E - kdz * zE) + kony * yE * (kmy - kdy),
		konx * (kdx * xE - x * E) + kony * (kdy * yE - y * E)]# + konz * (kdz * zE - z * E)]
	return f



def callkodefunc(t, m, e0, s0, konx, kony, kdx, kdy, kmx, kmy):
	abserr = 1.0e-12
	relerr = 1.0e-10
	#pvec = [konx, kony, konz, kdx, kdy, kdz, kmx, kmy]
	pvec = [konx, kony, kdx, kdy, kmx, kmy]
	#w0 = [s0, 0, 0, 0, 0, 0, e0]
	w0 = [s0, 0, 0, 0, e0]
	wsol = odeint(kdode, w0, t, args=(pvec,),atol=abserr, rtol=relerr)
	return wsol[:,3] * m * kony * (kmy - kdy)

#This is the objective function that lmfit can play with. Compute the deviation from the luminance for all potential datapoints.
def objective(params, data):
	stime = time.time()
	returnvalue = []
	c = 0
	m = params['M'].value
	kdx = params['Kdx'].value
	kdy = params['Kdy'].value
	#kdz = params['Kdz'].value
	kmx = params['Kmx'].value
	kmy = params['Kmy'].value
	konx = params["Konx"].value
	kony = params["Kony"].value
	#konz = params["Konz"].value
	for k in sorted(data.keys()):
		d = data[k]
		t = d[0]
		for j in range(1,len(d)):
			l = d[j]
			e0 = params['E_%d' % c].value
			s0 = params['s_%d' % c].value
			#fit = callkodefunc(t, m, e0, s0, konx, kony, konz, kdx, kdy, kdz, kmx, kmy)
			fit = callkodefunc(t, m, e0, s0, konx, kony, kdx, kdy, kmx, kmy)
			returnvalue.append(fit - l)
			c += 1
	retval = np.concatenate(returnvalue)
	print("Objective RMSE = ", np.sqrt(np.mean(np.square(retval))), ((time.time()-stime)), len(retval))
	return retval

#New ODE, that accounts for product inhibition at two different steps.
#x is the d-luciferin concentratin
#y is luciferyl-adenylate 
#z is oxyluciferin
#E is the enzyme
#I'm assuming that light emission happens in the enzyme catalyzed step when yE is converted to zE
#Same as above, but now I'm inputting explicit Kd from the literature to maybe make this fit not as bananas to deal with.
#Kd is koff/kon, so all the "koff" terms can be replaced by Kd * kon
#kcat is related to Km as well. Km = koff + kcat/kon

def permeabilityode(wvec, t, pvec):
	#xo, x, xE, yo, y, yE, zo, z, zE, E = wvec # current values for the concentrations
	xo, x, xE, yo, y, yE, E = wvec # current values for the concentrations
	#px, py,pz,R, konx, kony, konz, kdx, kdy, kdz, kmx, kmy = pvec
	px, py,R, konx, kony, kdx, kdy, kmx, kmy = pvec
	factor = 3/19
	f = [R * px * factor * (x-xo), #x0
		konx * (kdx * xE - x * E) + px * factor * (xo-x), #x
		konx * (x * E - kmx * xE), #xE
		R * py * factor * (y-yo), #y0
		kony * (kdy * yE - y * E) + py * factor * (yo-y), #y
		kony * (y * E - kmy * yE) + konx * (kmx - kdx) * xE, #yE
		#R * pz * factor * (z-zo), #z0
		#konz * (kdz * zE - z * E) + pz * factor * (zo-z), #z
		#konz * (z * E - kdz * zE) + kony * yE * (kmy - kdy), #zE
		#konx * (kdx * xE - x * E) + kony * (kdy * yE - y * E) + konz * (kdz * zE - z * E)]#E
		konx * (kdx * xE - x * E) + kony * (kdy * yE - y * E)]#E
	return f



def callpermeabilityodefunc(t, m, e0, s0, konx, kony, kdx, kdy, kmx, kmy, px,py,R):
	abserr = 1.0e-12
	relerr = 1.0e-10
	#pvec = [px,py,pz,R,konx, kony, konz, kdx, kdy, kdz, kmx, kmy]
	pvec = [px,py,R,konx, kony, kdx, kdy, kmx, kmy]
	#w0 = [s0, 0, 0, 0, 0, 0, 0, 0, 0, e0]
	w0 = [s0, 0, 0, 0, 0, 0, e0]
	wsol = odeint(permeabilityode, w0, t, args=(pvec,),atol=abserr, rtol=relerr)
	return R * wsol[:,5] * m * kony * (kmy - kdy)

def objective2(params, data):
	returnvalue = []
	stime = time.time()
	c = 0
	m = params['M'].value
	kdx = params['Kdx'].value
	kdy = params['Kdy'].value
	#kdz = params['Kdz'].value
	kmx = params['Kmx'].value
	kmy = params['Kmy'].value
	#This is noshell data. Doesn't involve P at all yet.
	konx = params["Konx"].value
	kony = params["Kony"].value
	#konz = params["Konz"].value
	for k in sorted(data['NoShell'].keys()):
		d = data['NoShell'][k]
		t = d[0]
		for j in range(1,len(d)):
			l = d[j]
			e0 = params['E_%d' % c].value
			s0 = params['s_%d' % c].value
			#fit = callkodefunc(t, m, e0, s0, konx, kony, konz, kdx, kdy, kdz, kmx, kmy)
			fit = callkodefunc(t, m, e0, s0, konx, kony, kdx, kdy, kmx, kmy)
			returnvalue.append(fit - l)
			c += 1
	#This is shell data. Needs P, calls a different ODE.
	for key in ['Capped', 'Uncapped']:
		px = params['px_%s' % key].value
		py = params['py_%s' % key].value
		#pz = params['pz_%s' % key].value
		konx = params["Konx"].value
		kony = params["Kony"].value
		#konz = params["Konz"].value
		for k in sorted(data[key].keys()):
			d = data[key][k]
			t = d[0]
			for j in range(1,len(d)):
				l = d[j]
				e0 = params['E_%d' % c].value
				s0 = params['s_%d' % c].value
				r = params['r_%d' % c].value
				#fit = callpermeabilityodefunc(t, m, e0, s0, konx, kony, konz, kdx, kdy, kdz, kmx, kmy, px, py, pz, r)
				fit = callpermeabilityodefunc(t, m, e0, s0, konx, kony, kdx, kdy, kmx, kmy, px, py, r)
				returnvalue.append(fit - l)
				c += 1
	retval = np.concatenate(returnvalue)
	print("Objective2 RMSE = ", np.sqrt(np.mean(np.square(retval))), ((time.time()-stime)))
	return retval

def refit(t, luminance, f, p0, bounds=(0,np.inf)):
	popt, pcov = curve_fit(f, t, luminance, bounds=bounds, p0=p0, maxfev=50000)
	# print(popt, f(t, *popt))
	# print(t, luminance)
	# exit()
	if len(popt) == 12:
		return (f(t, *popt), (popt[9],popt[10]))
	return f(t, *popt)

def calcrmse(data, fit):
	return np.sqrt(np.mean(np.square(data - fit)))

def plotfit(x, y, params, c, soln, key, title):
	fig, ax = plt.subplots(1,1,figsize=(8,6))
	ax.plot(x, y, label="Data", linewidth=1, color='k')
	m = params['M'].value
	kdx = params['Kdx'].value
	kdy = params['Kdy'].value
	#kdz = params['Kdz'].value
	kmx = params['Kmx'].value
	kmy = params['Kmy'].value
	konx = params["Konx"].value
	kony = params["Kony"].value
	#konz = params["Konz"].value
	e0 = params['E_%d' % c].value
	s0 = params['s_%d' % c].value
	if key != "NoShell":
		px = params['px_%s' % key].value
		py = params['py_%s' % key].value
		#pz = params['pz_%s' % key].value
		r = params['r_%d' % c].value
		fit = callpermeabilityodefunc(x, m, e0, s0, konx, kony, kdx, kdy, kmx, kmy, px, py, r)
		p = [m, e0, s0, konx, kony, kdx, kdy, kmx, kmy, px, py, r]
		fit2, pvals = refit(x, y, callpermeabilityodefunc, p)
		fout = open("specificps.csv", "a")
		fout.write("%s, %s, %g, %g\n" % (soln, key, pvals[0], pvals[1]))
		fout.close()
	else:
		fit = callkodefunc(x, m, e0, s0, konx, kony, kdx, kdy, kmx, kmy)
		p = [m, e0, s0, konx, kony, kdx, kdy, kmx, kmy]
		fit2 = refit(x, y, callkodefunc, p)
	ax.plot(x, fit, label="General Fit", linewidth=1, color='r')
	ax.plot(x, fit2, label="Specific Fit", linewidth=1, color='b')
	ax.set_ylim(bottom=0)
	ax.set_xlim(0, np.amax(x))
	ax.legend()
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Luminance")
	ax.set_title(title)
	ax.text(0.2,0.11, "General RMSE = %.2f" % (calcrmse(y, fit)), transform=ax.transAxes)
	ax.text(0.2,0.01, "Specific RMSE = %.2f" % (calcrmse(y, fit2)), transform=ax.transAxes)
	return fig
def plotall(soln, optp, datadict):
	counter = 0
	for key in ['NoShell','Capped', 'Uncapped']:
		for k in sorted(datadict[key].keys()):
			data = datadict[key][k]
			for j in range(1,len(data)):
				title = "%s %s %d" % (soln, key, counter)
				fig = plotfit(data[0], data[j], optp, counter, soln, key, title)
				fig.savefig("fits/%s-%s-%d.png" % (soln, key, counter))
				counter += 1
	
	# axin = ax.inset_axes([0.15, 0.52, 0.43, 0.43])
	# ax.plot(t, luminance, "k", linestyle = "-", label="Data")
	# axin.plot(t, luminance, 'k', label='Data')
	# axes = [ax, axin]
	# plotfit(t, func2, opts[0], "Brownian", "blue",axes)
	# plotfit(t, func, opts[1], "SqrtBrownian", "green",axes)
	# plotfit(t, func3, opts[2], "MM", "violet",axes)
	# plotfit(t, callodefunc, opts[3], "ODE", "orange",axes)
	# plotfit(t, callmultfunc, opts[4], "Mult", "red",axes)
	# axin.set_xlim(0,1)
	# ax.set_ylim(bottom=0)
	# ax.set_xlim(0,60)
	# axin.set_ylim(bottom=0)
	# ax.legend()
	# ax.indicate_inset_zoom(axin,zorder=-100)
	# ax.set_xlabel("Time (s)")
	# ax.set_ylabel("Luminance")
	# ax.set_title(title)
	# return fig

def filltable(params):
	table = np.empty(9, dtype=float)
	for i, l in enumerate(['x','y']):
		table[0+i] = params['p%s_Capped' % l].value
	#print(params['paccel'].value)
	table[2] = params['paccel'].value
	for i, l in enumerate(['x','y']):
		table[3+i] = params["Kd%s" % l].value
	for i, l in enumerate(['x','y']):
		table[5+i] = params["Km%s" % l].value
	for i, l in enumerate(['x','y']):
		table[7+i] = params["Kon%s" % l].value
	return table
def fillstderrtable(params):
	table = np.empty(9, dtype=float)
	for i, l in enumerate(['x','y']):
		table[0+i] = params['p%s_Capped' % l].stderr
	#print(params['paccel'].stderr)
	table[2] = params['paccel'].stderr
	for i, l in enumerate(['x','y']):
		table[3+i] = params["Kd%s" % l].stderr
	for i, l in enumerate(['x','y']):
		table[5+i] = params["Km%s" % l].stderr
	for i, l in enumerate(['x','y']):
		table[7+i] = params["Kon%s" % l].stderr
	print(table)
	return table

def writetable(table, residuals):
	parameterlist = ['$P_x$ (nm/s)','$P_y$ (nm/s)', 'Uncapped acceleration', '$K_d^x$ (\\unit{\\micro\\mole\\per\\L})', '$K_d^y$ (\\unit{\\micro\\mole\\per\\L})', '$K_m^x$ (\\unit{\\micro\\mole\\per\\L})', '$K_m^y$ (\\unit{\\micro\\mole\\per\\L})', '$K_{on}^x$ (\\unit{\\per\\s})', '$K_{on}^y$ (\\unit{\\per\\s})']
	fout = open("table.tex", "w")
	fout.write(r'''\begin{tabular}{lrrr}
\toprule
Parameter & \multicolumn{3}{c}{Solution} \\
 & Buffer & PEG 10\% & PEG 20\% \\
\midrule
''')
	for i in range(len(parameterlist)):
		s = parameterlist[i]
		for j in range(3):
			if np.isfinite(table[1][i][j]):
				s += " & \\num{{{0:f} \\pm {1:f}}}".format(table[0][i][j], table[1][i][j])
			else:
				s += " & \\num{{{0:.3g}}}".format(table[0][i][j])
		s += " \\\\\n"
		fout.write(s)
	fout.write("\\midrule\nRMSE & %.2f & %.2f & %.2f \\\\\n" % (residuals[0], residuals[1], residuals[2]))
	fout.write(r'''\bottomrule
\end{tabular}
''')
	fout.close()


#Load in the result from parsedata.py
file = open('alldata.pkl', 'rb')
# load info from file
alldata = pickle.load(file)
file.close()
#The table I want to make has 3 columns (Buffer, PEG 10%, PEG 20%)
#The table has Px, Py, Pz, Uncapped Acceleration, Kdx, Kdy, Kdz, Kmx, Kmy, Konx, Kony, Konz rows
tableplaceholder = np.zeros((2,9,3), dtype=float)
residuals = []
fout = open("specificps.csv", "w")
fout.close()
for j, soln in enumerate(['Buffer', 'PEG10', 'PEG20']):
	fout = open("lmfitresult-%s.pkl" % soln, "rb")
	result = pickle.load(fout)
	optparameters = result.params
	fout.close()
	plotall(soln, optparameters, alldata[soln])
	tableplaceholder[0,:,j] = filltable(optparameters)
	tableplaceholder[1,:,j] = fillstderrtable(optparameters)
	residuals.append(np.sqrt(np.mean(np.square(result.residual))))
writetable(tableplaceholder, residuals)
exit()
files = glob.glob("odsfiles/*ods")
for f in files:
	date = f[9:].split()[0]
	rmsedata = np.load("rmsedata-%s.npy" % date)
	fin = open("optdata-%s.pkl" % date, "rb")
	optdict = pickle.load(fin)
	fin.close()
	file = "./data.ods"

	try:
		df = read_ods(f, 2)
	except:
		df = read_ods(f, 1)
	print(df)
	columnheaders = df.columns.tolist()

	data = df.to_numpy()
	t = data[:,0]
	for i in range(1,data.shape[1]):
		l = data[:,i]
		opts = optdict[columnheaders[i]]
		fig = plotall(t, l, opts, columnheaders[i].strip())
		fig.savefig("fits/%s-%s.png" % (columnheaders[i].strip(), date))