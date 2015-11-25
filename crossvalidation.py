import pandas as pd
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
loansData['Interest.Rate'] = cleanInterestRate
cleanLoanLength = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))
loansData['Loan.Length'] = cleanLoanLength
loansData['FICO.Score'] = [int(val.split('-')[0]) for val in loansData['FICO.Range']]

import matplotlib.pyplot as plt

import numpy as np
from sklearn.cross_validation import KFold
import pandas as pd
import statsmodels.api as sm

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

y = np.matrix(intrate).transpose()
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

x = np.column_stack([x1,x2])

kf = KFold(10,n_folds=10)
for train, test in kf:
	print("%s %s" % (train, test))

X = np.array([[0., 0.], [1.,1.], [2.,2.], [3.,3.], [4.,4.], [5.,5.], [6.,6.], [7.,7.], [8.,8.], [9.,9.]])
y = np.array([0,1,2,3,4,5,6,7,8,9])
X_train, X_test, y_train, y_test = x[train], X[test], y[train], y[test]
	
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

print f.summary()

