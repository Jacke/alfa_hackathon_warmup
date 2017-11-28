from pandas_datareader import data as pdr
import pandas as pd

import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)

# download dataframe
data = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30")

from numpy import *
import datetime
import matplotlib.pylab as plt
import numpy as np
import statsmodels.api as sm
from collections import defaultdict
import pdb

# define method for pulling Adj Close from Yahoo! Finance
def Stock_Close(Ticker, YYYY, m, dd):
	start_date = datetime.datetime(YYYY, m, dd)
	pull = pdr.get_data_yahoo(Ticker, start = start_date) 
	close =  pull["Adj Close"]

	return close

# define method for pulling beta from Yahoo! Finance
def Stock_beta(Ticker, YYYY, m, dd):
	start_date = datetime.datetime(YYYY, m, dd)
	pull = pdr.get_data_yahoo(Ticker, start = start_date) 
	beta = pull["beta"]

	return beta 

# define method for pulling GDP from FRED
def Econ_env(YYYY, m, dd):	
	start_date = datetime.datetime(YYYY, m, dd)
	GDP = pdr.DataReader('GDP', "fred", start=start_date)
	sp500 = pdr.get_data_yahoo('^GSPC', start=start_date)

	Array = pdr.DataFrame({'S&P':sp500["Adj Close"]})

	return Array

def regression(DataFrame, assets, back, forward):
	window = defaultdict(list)
	regr = defaultdict(list)
	projection = {}
	profit = {}

	days = back*30 
	out = forward*30

	for ass in assets:
		for i in range(len(DataFrame.index) -1 , len(DataFrame.index) - days , -1):
			window[ass].append(DataFrame[ass][i])

		print("error will be here: >>>>>>")
		print(window[ass])
		print(range(0, len(window[ass])))
		
		X = range(0, len(window[ass]))
		X = sm.add_constant(X)

		temp = sm.OLS(window[ass], X )
		print("temp")
		print(temp)

		reg = temp.fit()
		regr[ass] = (reg.params)

		projection[ass] = window[ass][1] + regr[ass]*30
		profit[ass] = projection[ass] - DataFrame[ass][len(DataFrame.index) - 1]

	return regr, window, projection, profit

def Stock_stats(dataFrame, weights, assets, lam):
	[m,n] = shape(dataFrame)
	std = dataFrame.std()
	corr = dataFrame.corr()
	STD = {}
	buyin = []
	current = []
	ewma = {}
	EWMA = {}
	# transform the dataFrame index to a series so regression will
	# work

	for ass in assets:
		temp= []
		j = 0
		
		for i  in range( (len(dataFrame.index)-1), 0, -1):
			temp.append(( pow((dataFrame[ass][i] - dataFrame[ass].mean()), 2)
			 * pow(lam, i)))
			j = j+1
	
		# buyin.append(dataFrame[ass][0] * weights[ass] )
		STD[ass] = [std[ass] * weights[ass]]
		# current.append([dataFrame[ass][m-1] * weights[ass]])
		# sum the squared difference and compute EWMA over time for each asset
		ewma[ass] = sqrt((1-lam) * sum(temp))
		# take the weighted average of each asset, store in list and sum
		EWMA[ass] = ewma[ass] * weights[ass]

		
	# # calculate EWMA as described by Minkah
	# for ass in assets:
	# 	for i  in range( (len(dataFrame.index)-1), 0, -1):
	# 		temp.append((pow(dataFrame[ass][i] - dataFrame[ass].mean(), 2) * pow(lam, i)))
		
	# 	ewma[ass] = sqrt((1-lam) * sum(temp))
	# 	# take the weighted average of each asset, store in list and sum
	# 	EWMA.append(ewma[ass] * weights[ass])

	# calculate the line regression for the profit
	# parse and use only the past X months 

	# calculate the line of best fit for EWMA for each asset 

	# calculate the EU using the tail side risks and mean values and volility
	
	return STD, EWMA

def DirichletDistro(num, sum):
	# creates a Dirchlet distribtion (D) with characteristics
	# sum(D) = sum and len(D) = 1 w/all numbers being 
	# pseudorandom

	D = random.dirichlet(ones(num), size=sum)
	return list(D.reshape(-1)) 		#convert to list

def utl(profit, EWMA, beta, delta, t): 
	return 1 - exp( - beta * (profit / sqrt(EWMA)) * pow(delta, t) )

def var(X, weights, assets):
	corr = X.corr()
	X = zeros((1, len(assets)))
	j = 0

	# first sort the weights dict by alphabet since corr matrix will be
	sort = sorted(weights.keys())

	for ass in sort:
		X[0,j] = weights[ass]
		j = j+1

	var = dot(X, corr.as_matrix())
	var = dot(var, X.transpose())

	return var

def main():
	# define the desired portfolio characteristics
	std_max = .2		# maximum standard deviation 
	MAX_ITERS = 200 	# max number of iterations
	lam = .94  			# exponential decay number 
	exit_date = 12  	# when you sell stocks (in months)
	window_begin = 6    # how far back you want to window reg. (in months)
	beta = .6			# personal risk adversion level
	delta = .99			# discount factor 

	assets = (['GOOGLE', 'APPLE', 'CAT', 'SPDR_GOLD', 'OIL',
	 'NATURAL_GAS', 'USD', 'GOLDMANSACHS', 'DOMINION'])

	print('Pulling data from Yahoo! Finance')
	# Pull data from Yahoo! Finance 
	GOOG= Stock_Close('GOOG', 2010, 1, 1)
	AAPL = Stock_Close('AAPL', 2010,1, 1)
	SP500 = Stock_Close('^GSPC', 2010, 1, 1)
	CAT = Stock_Close('CAT', 2010, 1, 1)
	GOLD = Stock_Close('GLD', 2010, 1, 1)
	GAS = Stock_Close('GAZ', 2010, 1, 1)
	OIL = Stock_Close('OIL', 2010, 1, 1)
	GS = Stock_Close('GS', 2010, 1, 1)
	DOM = Stock_Close('D', 2010, 1, 1)
	
	# FX currency
	USD = Stock_Close('UUP', 2010, 1, 1)

	# create a dataframe housing the above
	X = pd.DataFrame({'GOOGLE':GOOG, 'APPLE':AAPL, 'CAT':CAT, 'SPDR_GOLD':GOLD,
	 'OIL':OIL, 'NATURAL_GAS':GAS, 'USD':USD, 'GOLDMANSACHS': GS, 'DOMINION':DOM})

	best = zeros(((4+len(assets)), MAX_ITERS))

	print('Running monte carlo simulation')
	for i in range(1, MAX_ITERS):
		print('Percent Done: \t' + str(float(i)/float(MAX_ITERS)*100)+' %')
		numWeight = DirichletDistro(len(assets),1)
		EU = {}
		
		# check to make sure sum weights = 1
		if int(sum(numWeight)) < 1.01: # account for floating point err
			#create dictionary of weights for each of the assets
			weights = dict(zip(assets, numWeight))

			STD, EWMA = Stock_stats(X, weights, assets, lam)

			# calculate price at future period 
			regr, window, projection, profit = regression(X, assets, window_begin, exit_date)
			# calculate expected utility
			for ass in assets:
				profit[ass] = profit[ass] * weights[ass]
				EU[ass] = utl(profit[ass], EWMA[ass], beta, delta, exit_date*30)
			
			# calculate the variance of the portfolio
			var_portfolio = var(X, weights, assets)

			# # check if the EWMA is above the specified limit
			if var_portfolio == std_max:
				best[:, i] = zeros(( (4+len(assets)) )) # drop the trial 
				# add more critera here...
			else:				# store trial profit, EWMA, STD in column of best
				best[0:4, i] = [sum(EU.values()), var_portfolio, sum(EWMA.values()), sum(profit.values())]
				sorted_assest = sorted(weights.keys())
				print(sum(profit.values()))
				j = 4
				for ass in sorted_assest:
					best[j, i] = weights[ass]
					j = j+1 

		else:
			print('Sum of weights does not equal 1')



	maxP = max(best[0,:]) 
	print( "The maximum profit to be made is: %f") % maxP
	# find the column where the sum is equal to the max
	opt = where(best[0,:] == maxP)
	# OptimalAllocation = dict(zip(assets, [float(xx) for xx in best[2:,opt]]))
	print('\n The optimal asset allocation in the portfolio is:')
	# print(OptimalAllocation)
	return X, best, assets, EU, regr, profit

if __name__ == "__main__":
    main()
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    #
    Y = [1054.209961, 1040.6099850000001, 1035.959961, 1034.48999, 1018.3800050000001, 1019.090027, 1032.5, 1020.909973, 1026.0, 1025.75, 1028.0699460000001, 1031.26001, 1039.849976, 1033.329956, 1025.900024, 1032.4799800000001, 1025.579956, 1025.5, 1016.6400150000001, 1017.1099849999999, 1019.2700199999999, 972.55999800000006, 973.330017, 970.53997799999991, 968.4500119999999, 988.2000119999999, 984.4500119999999, 992.80999800000006, 992.17999299999997, 992.0, 989.67999299999997, 987.830017, 989.25, 972.59997599999997, 977.0, 978.89001500000006, 969.96002200000009, 951.67999299999997, 957.78997799999991, 953.27001999999993, 959.10998499999994, 949.5, 944.48999000000003, 924.85998499999994, 920.9699710000001, 928.5300289999999, 932.4500119999999, 931.580017, 921.80999800000006, 915.0, 920.28997799999991, 925.10998499999994, 935.09002699999996, 932.07000700000003, 929.080017, 926.5, 935.9500119999999, 927.80999800000006, 928.4500119999999, 937.34002699999996, 939.330017, 929.57000700000003, 921.28997799999991, 913.80999800000006, 915.89001500000006, 921.2800289999999, 927.0, 924.69000199999994, 906.65997300000004, 910.669983, 910.97998000000007, 926.96002200000009, 922.2199710000001, 922.669983, 914.39001500000006, 907.23999000000003, 922.90002400000003, 926.78997799999991, 929.35998499999994, 927.96002200000009, 923.65002400000003, 930.39001500000006, 930.830017, 930.5, 941.5300289999999, 934.09002699999996, 947.7999880000001, 950.7000119999999, 980.34002699999996, 972.919983, 968.15002400000003, 970.89001500000006, 965.40002400000003, 953.419983, 955.98999000000003, 947.15997300000004, 943.830017, 930.09002699999996, 928.7999880000001, 918.59002699999996, 906.69000199999994, 911.71002200000009, 898.7000119999999, 908.72998000000007, 917.78997799999991, 940.48999000000003, 927.330017, 952.27001999999993, 965.59002699999996, 957.09002699999996, 959.4500119999999, 950.6300050000001, 957.3699949999999, 939.7800289999999, 942.30999800000006, 950.76000999999997, 953.40002400000003, 942.90002400000003, 949.830017, 983.40997300000004, 980.94000199999994, 976.57000700000003, 983.67999299999997, 975.59997599999997, 966.9500119999999, 964.85998499999994, 975.8800050000001, 971.4699710000001, 969.53997799999991, 954.96002200000009, 948.82000700000003, 941.85998499999994, 934.01000999999997, 930.23999000000003, 919.6199949999999, 943.0, 937.080017, 932.2199710000001, 930.59997599999997, 928.7800289999999, 932.169983, 934.2999880000001, 927.1300050000001, 931.65997300000004, 927.03997799999991, 916.44000199999994, 912.57000700000003, 905.96002200000009, 874.25, 871.72998000000007, 872.2999880000001, 862.76000999999997, 843.19000199999994, 841.65002400000003, 838.21002199999998, 836.82000700000003, 837.169983, 823.55999800000006, 824.32000700000003, 823.34997599999997, 824.72997999999995, 824.669983, 827.88000499999998, 831.40997300000004, 834.57000700000003, 838.54998799999998, 829.55999800000006, 831.5, 831.40997300000004, 820.919983, 819.51000999999997, 814.42999299999997, 817.580017, 829.59002699999996, 830.46002199999998, 848.40002400000003, 852.1199949999999, 848.7800289999999, 847.2000119999999]
    X = range(0,179)
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    results.params
    results.tvalues
    print(results.t_test([1, 0]))
    print(results.f_test(np.identity(2)))
    print(model)


