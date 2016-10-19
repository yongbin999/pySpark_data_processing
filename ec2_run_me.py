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
	rdd_train_converted = rdd_train_converted.map(lambda arr:  [int(arr[0])] +map(float,arr[1:]) ).cache()

	rdd_train_1 = rdd_train_converted.map(lambda arr: np.append(arr, [1]) ).cache()
	x_dot_y = rdd_train_1.map(lambda array: np.dot(np.array(array[1:]).transpose(),array[0]))
	x_dot_y_sum = x_dot_y.reduce(lambda a,b:a+b)

	##q2 <<- transpose a matrix in spark?
	xfeatures_matrix = rdd_train_1.map(lambda arr: np.outer(arr[1:],arr[1:]))
	xfeatures_matrix = xfeatures_matrix.reduce(lambda a,b : a+b)

	##q3 << how to invert without collect?
	from numpy.linalg import inv
	invmatrix = inv(xfeatures_matrix)
	feature_weights = np.dot(invmatrix,x_dot_y_sum)

	rdd_test_converted = rdd_test.map(lambda arr: np.array(arr.split(',')))
	rdd_test_converted = rdd_test_converted.map(lambda arr:  [int(arr[0])] +map(float,arr[1:]) )

	##q4 append 1 to end?
	rdd_test_1 = rdd_test_converted.map(lambda arr: np.append(arr, [1]) ).cache()
	test_target = rdd_test_1.map(lambda arr: int(arr[0]))
	test_features = rdd_test_1.map(lambda arr: arr[1:])
	predict = test_features.map(lambda feat : np.dot(np.array(feature_weights).transpose(),feat))


	test_target_predict = sc.parallelize(zip(test_target.collect(), predict.collect())).cache()
	mean_abs_error = test_target_predict.map(lambda pair: abs(pair[0]-pair[1])).reduce(lambda a,b: a+b)/test_target_predict.count()
	print "q2.4 MAE score "
	print mean_abs_error
	print





	#==================================
    #End of your linear regression code 
    #==================================

	times.append([i,time.time()-start])

	print times

	sc.stop()






