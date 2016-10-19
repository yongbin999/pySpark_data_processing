from pyspark import SparkContext, SparkConf
import numpy as np
import time
#============
#Add any extra imports here
#============

with open ("/root/spark-ec2/cluster-url", "r") as myfile:
	master=myfile.readline().strip()

times=[]
for i in range(1,9):

	conf = (SparkConf()
	         .set("spark.cores.max", i))

	sc = SparkContext(master, "ec2_run_me", conf=conf)
	sc.setLogLevel("ERROR")

	start=time.time()

	rdd_train = sc.textFile("/songs/train_data.txt")
	rdd_test = sc.textFile("/songs/test_data.txt")

	#==================================
	#Start of your linear regression code
	#==================================
	print "start running"
	rdd_train_converted = rdd_train.map(lambda arr: np.array(arr.split(',')))
	rdd_train_converted = rdd_train_converted.map(lambda arr:  [int(arr[0])] +map(float,arr[1:]) )
	rdd_train_1 = rdd_train_converted.map(lambda arr: np.append(arr, [1]) ).cache()


	eucleadian_norm = []
	for x in range(0,len(rdd_train_1.take(1)[0])):
		if x==0:
			eucleadian_norm += [1]
		else:
			eucleadian_norm += [pow(rdd_train_1.map(lambda arr: pow(arr[x],2) ).sum() , 0.5)]

	normalized = rdd_train_1.map(lambda arr: [ value / eucleadian_norm[index] for index,value in enumerate(arr) ]) #<<<<<


	x_dot_y = normalized.map(lambda array: np.dot(np.array(array[1:]).transpose(),array[0]))
	x_dot_y_sum = x_dot_y.reduce(lambda a,b:a+b)
	#nump.
	xfeatures_matrix = normalized.map(lambda arr: np.outer(arr[1:],arr[1:]))
	xfeatures_matrix = xfeatures_matrix.reduce(lambda a,b : a+b)


	from numpy.linalg import inv
	invmatrix = inv(xfeatures_matrix)
	feature_weights = np.dot(invmatrix,x_dot_y_sum)


	## test dataset
	rdd_test_converted = rdd_test.map(lambda arr: np.array(arr.split(',')))
	rdd_test_converted = rdd_test_converted.map(lambda arr:  [int(arr[0])] +map(float,arr[1:]) )
	rdd_test_1 = rdd_test_converted.map(lambda arr: np.append(arr, [1]) ).cache()

	test_target = rdd_test_1.map(lambda arr: int(arr[0])).collect()
	
	test_features = rdd_test_1.map(lambda arr: arr[1:])
	test_features = test_features.map(lambda arr: [ value / eucleadian_norm[index+1] for index,value in enumerate(arr) ])  #<<<<<<

	predict = test_features.map(lambda feat : np.dot(np.array(feature_weights).transpose(),feat)).collect()

	for index, x in enumerate(predict):
		if x <1930:
			predict[index] = 1930
		if x >2010:
			predict[index] = 2010


	test_target_predict = sc.parallelize(zip(test_target, predict)).cache()
	mean_abs_error = test_target_predict.map(lambda pair: abs(pair[0]-pair[1])).reduce(lambda a,b: a+b)/test_target_predict.count()
	print "features norm + output norm 's  MAE score "
	print mean_abs_error
	print





	#==================================
	#End of your linear regression code 
	#==================================

	times.append([i,time.time()-start])

	print times

	sc.stop()






