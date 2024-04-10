#This takes all the ods files from the odsfiles directory, and puts them into a single pickle that is quick to load.
import pandas as pd
import glob 
import pickle
files = glob.glob("odsfiles/*ods")
alldata = dict()
alldata["PEG10"] = dict()
alldata["PEG20"] = dict()
alldata["Buffer"] = dict()
for k in alldata.keys():
	alldata[k]["NoShell"] = dict()
	alldata[k]["Capped"] = dict()
	alldata[k]["Uncapped"] = dict()
for f in files:
	ods = pd.ExcelFile(f)
	date = f[9:].split()[0]
	for sheet in ods.sheet_names:
		df = pd.read_excel(f, sheet_name=sheet)
		print(sheet, df)
		datesheet = date + "/" + sheet
		columnheaders = df.columns.tolist()
		data = df.to_numpy()
		t = data[:,0]
		for i in range(1,data.shape[1]):
			#Pick which conditions this was in.
			if "Shell" in columnheaders[i]:
				#print("This is a measurement taken without a shell!")
				key1 = "NoShell"
			elif "Capped" in columnheaders[i]:
				#print("This measurement taken in a capped shell")
				key1 = "Capped"
			elif "Uncapped" in columnheaders[i]:
				#print("This measurement taken in an uncapped shell")
				key1 = "Uncapped"
			else:
				print("Something went wrong here.")
				print(f, columnheaders[i])
				exit()
			if ("PEG10" in columnheaders[i] or "PEG10" in sheet):
				datatype = "PEG10"
			elif ("PEG" in columnheaders[i] or "PEG" in sheet):
				datatype = "PEG20"
			else:
				datatype = "Buffer"
			print(columnheaders[i], sheet, key1, datatype)
			l = data[:,i]
			if datesheet not in alldata[datatype][key1].keys():
				alldata[datatype][key1][datesheet] = [t]
			alldata[datatype][key1][datesheet].append(l)
fout = open("alldata.pkl", "wb")
pickle.dump(alldata,fout)
fout.close()
