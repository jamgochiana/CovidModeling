'''
This file contains the functions that take in the data 
with hospitalizations and deaths, as well as the estimated
gamma parameters, and make predictions into the future 
regarding hospitalizations and deaths
'''


def gamma4_predict_hosp(U, V, pred_gammas, forward, delta1, delta2):
	'''
	Function that predicts hospitalizations several days ahead
	assuming the 4-gamma infection dynamics model. 

	U - (np array) estimated (computed) number of daily
	infections of U-type individuals

	V - (np array) number of daily infections of V-type 
	individuals (this data us given at the beginning)

	pred - (np array) size = (T, 4)

	forward: (int) how many days we want to predict forward 
	into the future 
	'''

	predictions = np.zeros(len(U))

	for t in range(T + int_inf + delta2, len(U) - delta1):
		gm = pred_gammas[t]
		nb = U[t - delta2] + V[t - delta2]
		Unext = U[t - delta2]
		Vnext = V[t - delta2]
		for i in range(forward):
			Vnext = gm[0] * Vnext + gm[1] * Unext
			Unext = gm[2] * Vnext + gm[3] * Unext
			nb = Vnext + Unext
		predictions[t + forward + delta1 - delta2] = Vnext

	return predictions

def gamma2_predict_hosp(U, V, pred, forward, delta1, delta2):
	'''
	Function that predicts hospitalizations several days ahead
	assuming the 2-gamma infection dynamics model. 

	U - (np array) estimated (computed) number of daily
	infections of U-type individuals

	V - (np array) number of daily infections of V-type 
	individuals (this data us given at the beginning)

	pred - (np array) size = (T, 2)

	forward: (int) how many days we want to predict forward 
	into the future 
	'''

	predictions = np.zeros(len(U))

	for t in range(T + int_inf + delta2, len(U) - delta1):
		gm = pred_gammas[t]
		nb = U[t - delta2] + V[t - delta2]
		for i in range(forward-1):
			nb = nb * (gm[0] + gm[1])
		Vnext = nb * gm[0]
		predictions[t + forward + delta1 - delta2] = Vnext

	return predictions




