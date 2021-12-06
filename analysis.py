#analysis of test data

import matplotlib.pyplot as plt

accuracyMultiNB = []
accuracySgdClass = []
accuracyPA = []

outputFile = open('output.txt', 'r')
lines = outputFile.readlines()
count = 0
xAxis = []
nums = ['0', '1']
labelList = []
for i in range(0, len(lines)):
	if i>2:		
		if count == 0:
			a = float(lines[i])
			accuracyMultiNB.append(a)
			count = 1
		elif count == 1:
			a = float(lines[i])
			accuracySgdClass.append(a)
			count = 2
		elif count == 2:
			a = float(lines[i])
			accuracyPA.append(a)
			count = 0
		
for i in range(0, len(accuracyMultiNB)):
	xAxis.append(i+1)
	

"""print("MNB: ", accuracyMultiNB)
print("SGD: ", accuracySgdClass)
print("PAC: ", accuracyPA)"""

plt.plot(xAxis, accuracyMultiNB, label="Multinomial Naive Bayes")
plt.plot(xAxis, accuracySgdClass, label="SGD Classifier")
plt.plot(xAxis, accuracyPA, label="Passive Aggressive Classifier")
plt.title('Multinomial Naive Bayes')
plt.xlabel('Test Batch no.')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()
