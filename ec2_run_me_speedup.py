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


	##q2 <<- speedup by doing only 1 full multiply
	full_matrix = rdd_train_1.map(lambda arr: np.outer(arr,arr[1:]))
	xfeatures_matrix = full_matrix.map(lambda arr: arr[1:]).reduce(lambda a,b:a+b)
	x_dot_y_sum = full_matrix.map(lambda arr: arr[0]).reduce(lambda a,b:a+b)


	##q3 invert and multiply
	from numpy.linalg import inv
	invmatrix = inv(xfeatures_matrix)
	feature_weights = np.dot(invmatrix,x_dot_y_sum)



	## test dataset
	rdd_test_converted = rdd_test.map(lambda arr: np.array(arr.split(',')))
	rdd_test_converted = rdd_test_converted.map(lambda arr:  [int(arr[0])] +map(float,arr[1:]) )
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






