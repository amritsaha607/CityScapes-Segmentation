from metrics import accuracy
import numpy as np

# Checks
def matchVal(y, y_):
	return 1 if ((y-y_)<=y_*1e-4) else 0


# Tests

def get_test_case_accuracy(test_id=1):

	y = np.zeros((5, 5))
	y_pred = y.copy()

	if test_id==1:
		y[1:4, 1:4] = 1
		y_pred[2:3, 2:3] = 1
		acc = 0.68

	if test_id==2:
		y[1, 4] = 1
		y_pred[2, 2] = 1
		acc = 0.92

	res = {
		'y': y,
		'y_pred': y_pred,
		'acc': acc,
	}
	return res

def test_accuracy(test_id=1):
	res = get_test_case_accuracy(test_id=test_id)
	y, y_pred, acc_ = res['y'], res['y_pred'], res['acc']
	acc = accuracy(y, y_pred)
	print(matchVal(acc, acc_))
	
def run(func, test_ids=1):
	if isinstance(test_ids, int):
		func(test_ids)
	elif isinstance(test_ids, list):
		for test_id in test_ids:
			func(test_id)
	else:
		print("test ids must be list or int, found type {}".format(str(type(test_ids))))