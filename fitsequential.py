#The idea behind this script is to run the model that we have 3 times, once for the buffer,
#once for 10% PEG, and once for 20% PEG. With any luck, the permeability coefficients determined will be similar enough, and give us some idea about variation.
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

def inhibitionode(wvec, t, pvec):
	x, xE, y, yE, z, zE, E = wvec # current values for the concentrations
	konx, kony, konz, koffx, koffy, koffz, kcat1, kcat2 = pvec
	f = [-konx * x * E + koffx * xE,
		konx * x * E - koffx * xE - kcat1 * xE,
		-kony * y * E + koffy * yE,
		kony * y * E - koffy * yE + kcat1 * xE - kcat2 * yE,
		-konz * z * E + koffz * zE,
		konz * z * E - koffz * zE + kcat2 * yE,
		-konx * x * E - kony * y * E - konz * z * E + koffx * xE + koffy * yE + koffz * zE]
	return f


def callinhibitodefunc(t, m, e0, s0, konx, kony, konz, koffx, koffy, koffz, kcat1, kcat2):
	abserr = 1.0e-12
	relerr = 1.0e-10
	pvec = [konx, kony, konz, koffx, koffy, koffz, kcat1, kcat2]
	w0 = [s0, 0, 0, 0, 0, 0, e0]
	wsol = odeint(inhibitionode, w0, t, args=(pvec,),atol=abserr, rtol=relerr)
	return wsol[:,3] * m * kcat2


#What happens if we take this to be a 1-step reaction?
#Then we don't get the initial slope up, so this model isn't correct.
def simpleinhibitionode(wvec, t, pvec):
	x, xE, z, zE, E = wvec # current values for the concentrations
	konx, konz, koffx, koffz, kcat1 = pvec
	f = [-konx * x * E + koffx * xE,
		konx * x * E - koffx * xE - kcat1 * xE,
		-konz * z * E + koffz * zE,
		konz * z * E - koffz * zE + kcat1 * xE,
		-konx * x * E - konz * z * E + koffx * xE + koffz * zE]
	return f
def callsimpleinhibitodefunc(t, m, e0, s0, konx, konz, koffx, koffz, kcat1):
	abserr = 1.0e-12
	relerr = 1.0e-10
	pvec = [konx, konz, koffx, koffz, kcat1]
	w0 = [s0, 0, 0, 0, e0]
	wsol = odeint(simpleinhibitionode, w0, t, args=(pvec,),atol=abserr, rtol=relerr)
	return wsol[:,1] * m * kcat1


def fit(t, luminance, f, p0, bounds=(0,np.inf)):
	#try:
		popt, pcov = curve_fit(f, t, luminance, bounds=bounds, p0=p0, maxfev=5000)
		return popt
	#except RuntimeError:
	#	return None

def plotfit(t, f, popt, label, color, axes):
	for a in axes:
		a.plot(t, f(t, *popt), color, label=label, linewidth=1)

def calcrmse(t,luminance,f,popt):
	if popt is None:
		return 9999
	return np.sqrt(np.mean(np.square(f(t, *popt) - luminance)))

def plotall(t, luminance, opt, title):
	fig, ax = plt.subplots(1,1,figsize=(8,6))
	axin = ax.inset_axes([0.35, 0.52, 0.43, 0.43])
	ax.plot(t, luminance, "k", linestyle = "-", label="Data")
	axin.plot(t, luminance, 'k', label='Data')
	axes = [ax, axin]
	plotfit(t, callkodefunc, opt, "Fit", "red",axes)
	axin.set_xlim(0,1)
	ax.set_ylim(bottom=0)
	ax.set_xlim(0,np.amax(t))
	axin.set_ylim(bottom=0)
	ax.legend()
	ax.indicate_inset_zoom(axin,zorder=-100)
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Luminance")
	ax.set_title(title)
	return fig

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
	#print("Objective RMSE = ", np.sqrt(np.mean(np.square(retval))), ((time.time()-stime)), len(retval))
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
	#print("Objective2 RMSE = ", np.sqrt(np.mean(np.square(retval))), ((time.time()-stime)))
	return retval

def runnewoptimization(soln, datadict):
	params = Parameters()

	params.add('Kdiffx', value=1.0, min=0)
	if soln == 'Buffer':
		#Fixed parameter based on https://doi.org/10.1039/b809935a
		params.add('Kmx', value=14.7, min=0, vary=False)
	else:
		params.add('Kmx', value=14.7, min=1.47, max=147)
	params.add('Kdx', value=13.7, min=0, expr='Kmx - Kdiffx')
	params.add('Kdiffy', value=1.0, min=0)
	if soln == 'Buffer':
		#Fixed parameter based on https://doi.org/10.1039/b809935a
		params.add('Kmy', value=14.9, min=0, vary=False)
	else:
		params.add('Kmy', value=14.9, min=1.49, max=149)
	params.add('Kdy', value=13.9, min=0, expr='Kmy - Kdiffy')
	#params.add('Kdz', value=1, min=0)
	params.add("M", value=1.0, min=0)
	counter = 0
	
	params.add("Konx", value=1.0, min=0)
	params.add("Kony", value=1.0, min=0)
	#params.add("Konz", value=1.0, min=0)
	fig, ax = plt.subplots(1,1)
	for k in sorted(datadict['NoShell'].keys()):
		data = datadict['NoShell'][k]
		for j in range(1,len(data)):
			ax.plot(data[0], data[j], color="C%d" % counter)
			params.add("E_%d" % counter, value=1, min=0)
			params.add("s_%d" % counter, value=1, min=0)
			counter += 1
	print(counter)
	fig.savefig("test-%s.png" % soln)
	start = time.time()
	thingtopass = dict()
	thingtopass['data'] = datadict["NoShell"]

	result = minimize(objective, params, kws=thingtopass)
	report_fit(result,show_correl=False)
	fout = open("lmfitresultnoshell-%s.pkl" % soln, "wb")
	pickle.dump(result,fout)
	fout.close()
	end = time.time()
	print("It took %f minutes" % ((end-start) / 60))

	#Now we optimize with permeability in place.
	params = result.params
	params.add("px_Capped", value=1, min=0)
	params.add("py_Capped", value=1, min=0)
	#params.add("pz_Capped", value=1, min=0)
	params.add('paccel',value=1,min=1)
	params.add("px_Uncapped", value=1, min=0, expr='px_Capped * paccel')
	params.add("py_Uncapped", value=1, min=0, expr='py_Capped * paccel')
	#params.add("pz_Uncapped", value=1, min=0, expr='pz_Capped * paccel')
	for key in ['Capped', 'Uncapped']:
		for k in sorted(datadict[key].keys()):
			data = datadict[key][k]
			for j in range(1,len(data)):
				params.add("E_%d" % counter, value=1, min=0)
				params.add("s_%d" % counter, value=1, min=0)
				params.add("r_%d" % counter, value=0.01, min=0, max=1)
				counter += 1
	start = time.time()

	thingtopass = dict()
	thingtopass['data'] = datadict

	result = minimize(objective2, params, kws=thingtopass)
	report_fit(result,show_correl=False)
	fout = open("lmfitresult-%s.pkl" % soln, "wb")
	pickle.dump(result,fout)
	fout.close()
	end = time.time()
	print("It took %f minutes" % ((end-start) / 60))
	return result

def reoptimize(soln, datadict, optparams):
	params = Parameters()

	params.add('Kdiffx', value=optparams['Kdiffx'].value, min=0, vary=False)
	if soln == 'Buffer':
		#Fixed parameter based on https://doi.org/10.1039/b809935a
		params.add('Kmx', value=14.7, min=0, vary=False)
	else:
		params.add('Kmx', value=optparams['Kmx'].value, min=1.47, max=147, vary=False)
	params.add('Kdx', value=optparams['Kmx'].value - optparams['Kdiffx'].value, min=0, expr='Kmx - Kdiffx', vary=False)
	params.add('Kdiffy', value=optparams['Kdiffy'].value, min=0, vary=False)
	if soln == 'Buffer':
		#Fixed parameter based on https://doi.org/10.1039/b809935a
		params.add('Kmy', value=14.9, min=0, vary=False)
	else:
		params.add('Kmy', value=optparams['Kmy'].value, min=1.49, max=149, vary=False)
	params.add('Kdy', value=optparams['Kmy'].value - optparams['Kdiffy'].value, min=0, vary=False, expr='Kmy - Kdiffy')
	#params.add('Kdz', value=1, min=0)
	params.add("M", value=optparams['M'].value, min=0)
	counter = 0
	
	params.add("Konx", value=optparams['Konx'].value, min=0, vary=False)
	params.add("Kony", value=optparams['Kony'].value, min=0, vary=False)
	#params.add("Konz", value=1.0, min=0)
	fig, ax = plt.subplots(1,1)
	for k in sorted(datadict['NoShell'].keys()):
		data = datadict['NoShell'][k]
		for j in range(1,len(data)):
			ax.plot(data[0], data[j], color="C%d" % counter)
			params.add("E_%d" % counter, value=1, min=0)
			params.add("s_%d" % counter, value=1, min=0)
			counter += 1
	print(counter)
	fig.savefig("test-%s.png" % soln)
	# start = time.time()
	# thingtopass = dict()
	# thingtopass['data'] = datadict["NoShell"]

	# result = minimize(objective, params, kws=thingtopass)
	# report_fit(result,show_correl=False)
	# fout = open("lmfitresultnoshell-%s.pkl" % soln, "wb")
	# pickle.dump(result,fout)
	# fout.close()
	# end = time.time()
	#print("It took %f minutes" % ((end-start) / 60))

	#Now we optimize with permeability in place.
	params.add("px_Capped", value=optparams['px_Capped'].value, min=0, vary=False)
	params.add("py_Capped", value=optparams['py_Capped'].value, min=0, vary=False)
	#params.add("pz_Capped", value=1, min=0)
	params.add('paccel',value=optparams['paccel'].value,min=1, vary=False)
	params.add("px_Uncapped", value=optparams['paccel'].value * optparams['px_Capped'].value, min=0, expr='px_Capped * paccel', vary=False)
	params.add("py_Uncapped", value=optparams['paccel'].value * optparams['py_Capped'].value, min=0, expr='py_Capped * paccel', vary=False)
	#params.add("pz_Uncapped", value=1, min=0, expr='pz_Capped * paccel')
	for key in ['Capped', 'Uncapped']:
		for k in sorted(datadict[key].keys()):
			data = datadict[key][k]
			for j in range(1,len(data)):
				params.add("E_%d" % counter, value=1, min=0)
				params.add("s_%d" % counter, value=1, min=0)
				params.add("r_%d" % counter, value=0.01, min=0, max=1)
				counter += 1
	start = time.time()

	thingtopass = dict()
	thingtopass['data'] = datadict
	#Minimize the first time, not allowing anything to change except E, s, and r
	result = minimize(objective2, params, kws=thingtopass)
	params = result.params
	for var in ['Kdiffx','Kmx','Kdx','Kdiffy','Kmy','Kdy','Konx','Kony','px_Capped','py_Capped','paccel','px_Uncapped','py_Uncapped']:
		params[var].vary = True
	#Minimize a second time allowing everything to shift.
	result = minimize(objective2, params, kws=thingtopass)
	report_fit(result,show_correl=False)
	fout = open("lmfitresult-%s.pkl" % soln, "wb")
	pickle.dump(result,fout)
	fout.close()
	end = time.time()
	print("It took %f minutes" % ((end-start) / 60))
	return result

#Load in the result from parsedata.py
file = open('alldata.pkl', 'rb')
# load info from file
alldata = pickle.load(file)
file.close()
soln = 'Buffer'
bufferresult = runnewoptimization(soln, alldata[soln])

# fin = open('lmfitresult-Buffer.pkl', 'rb')
# result = pickle.load(fin)
# fin.close()

for soln in ['PEG10', 'PEG20']:
	optparams = bufferresult.params
	reoptimize(soln, alldata[soln], optparams)







exit()