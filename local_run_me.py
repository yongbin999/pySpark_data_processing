from pyspark import SparkContext
import numpy as np
from numpy.linalg import inv
import time
#============
#Add any extra imports here
## /vagrant/spark-1.6.1-bin-hadoop2.6/bin/pyspark --master local[4]
## /vagrant/spark-1.6.1-bin-hadoop2.6/bin/pyspark
#============

sc = SparkContext("local", "local_run_me")
sc.setLogLevel("ERROR")

rdd_train = sc.textFile("/vagrant/shared_files/HW_Assignments/HW03/Data/Songs/train_data.txt")
rdd_test = sc.textFile("/vagrant/shared_files/HW_Assignments/HW03/Data/Songs/test_data.txt")

#==================================
#Start of your code
#==================================

rdd_train_converted = rdd_train.map(lambda arr: np.array(arr.split(',')))
rdd_train_converted = rdd_train_converted.map(lambda arr:  [int(arr[0])] +map(float,arr[1:]) ).cache()

## instead of manual split what lib to split

## q1
print "num of datarow:", rdd_train_converted.count()
##q2
pair = rdd_train_converted.map(lambda arr: (1,arr[0]))
sumyear = pair.reduceByKey(lambda a,b : a+b).collect()
avg = sumyear[0][1]/float(rdd_train_converted.count())
print "avg year: ", avg

##q3
ytarget = rdd_train_converted.map(lambda arr: arr[0])
innerterm = ytarget.map(lambda arr: pow((arr-avg),2)).reduce(lambda a,b : a+b)
sam_std = pow( innerterm / (ytarget.count()-1)  ,0.5)
#stdev = ytarget.stdev()
print "std : ", sam_std

##q4
minx = int(ytarget.min())
maxx = int(ytarget.max()+1)

pair = ytarget.map(lambda year: (year,1) )
years_count = pair.reduceByKey(lambda a,b : a+b)
results = years_count.collect()
##results.sort(key=lambda x: x[0])
##fill in blank for empty years
results = dict(results)
hist = [results.get(x) for x in range(minx,maxx)]
for i in range(0,len(hist)):
	if (hist[i]==None):
		hist[i] = 0

bins = [x for x in range(minx,maxx)]
##reduce the label to every 10 years
for i in range(0,len(bins)):
	if (bins[i]%10!=0):
		bins[i] = ''


import matplotlib.pyplot as plt
plt.bar(range(len(bins)),hist, align="center");
plt.gca().set_xticks(range(len(hist)+1))
plt.gca().set_xticklabels(bins[:])
plt.xlabel('Years from 1930-2010')
plt.ylabel('Number of Songs')
plt.title('Distribution of Songs by year')

#plt.show()

##q5
f1_dot_f2 = rdd_train_converted.map(lambda arr: np.dot(arr[1],arr[2])).reduce(lambda x,y: x+y)
print "feat.1 dot feat.2:", f1_dot_f2
print


## part2

start=time.time()
##q1
rdd_train_1 = rdd_train_converted.map(lambda arr: np.append(arr, [1]) ).cache()
#rdd_train_1 = rdd_train_1.map(lambda array: np.array(map(float, array)) )

x_dot_y = rdd_train_1.map(lambda array: np.dot(np.array(array[1:]).transpose(),array[0]))
x_dot_y_sum = x_dot_y.reduce(lambda a,b:a+b)
print "q2.1 first 4 of x_dot_y_sum: "
print x_dot_y_sum[:4]
print


##q2 <<- transpose a matrix in spark?
#nump.
xfeatures_matrix = rdd_train_1.map(lambda arr: np.outer(arr[1:],arr[1:]))
xfeatures_matrix = xfeatures_matrix.reduce(lambda a,b : a+b)

##speedup
full_matrix = rdd_train_1.map(lambda arr: np.outer(arr,arr[1:]))
xfeatures_matrix = full_matrix.map(lambda arr: arr[1:]).reduce(lambda a,b:a+b)
x_dot_y_sum = full_matrix.map(lambda arr: arr[0]).reduce(lambda a,b:a+b)


print "q2.2 weight matrix"
for i in range(0,4): 
	for j in range(0,4): 
		print xfeatures_matrix[i][j],
	print


##q3 << how to invert without collect?
invmatrix = inv(xfeatures_matrix)
feature_weights = np.dot(invmatrix,x_dot_y_sum)

#invmatrix = sc.parallelize(invmatrix)
#feature_weights = invmatrix.map(lambda row : np.dot(row, x_dot_y_sum)).collect()

print "q2.3 feat weights: "
print feature_weights[:4]
print

rdd_test_converted = rdd_test.map(lambda arr: np.array(arr.split(',')))
rdd_test_converted = rdd_test_converted.map(lambda arr:  [int(arr[0])] +map(float,arr[1:]) )

##q4 append 1 to end?
rdd_test_1 = rdd_test_converted.map(lambda arr: np.append(arr, [1]) ).cache()
test_target = rdd_test_1.map(lambda arr: int(arr[0]))
test_features = rdd_test_1.map(lambda arr: arr[1:])
predict = test_features.map(lambda feat : np.dot(np.array(feature_weights).transpose(),feat))
print "q2.4 predicts: "
print predict.collect()[:4]
print

predict_rounded = predict.map(lambda predict: int(round(predict)))
predict_rounded_results = predict_rounded.collect()
print predict_rounded_results[:4]
print


test_target_predict = sc.parallelize(zip(test_target.collect(), predict.collect())).cache()
mean_abs_error = test_target_predict.map(lambda pair: abs(pair[0]-pair[1])).reduce(lambda a,b: a+b)/test_target_predict.count()
print "q2.4 MAE score "
print mean_abs_error
print

print "time used: ",time.time()-start
print


"""
## added .cacheing above to improve proformance



## extra codes to improve the accuracy
## feature standardization
## x' = (x - xavg ) / xstd

"""

start=time.time()

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



rdd_test_converted = rdd_test.map(lambda arr: np.array(arr.split(',')))
rdd_test_converted = rdd_test_converted.map(lambda arr:  [int(arr[0])] +map(float,arr[1:]) )
##q4 append 1 to end?
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
print "time used: ",time.time()-start

""" std features
[10.663468283818347, 5.835219620320669, 49.173103164265122, 34.330924461019542, 15.852988792844851, 
22.627626529403631, 13.396459537433167, 13.692569836714354, 7.83781388408193, 10.059905470501301, 
6.4345347450531181, 4.2552456266402503, 8.1029234103034202, 22.514395917154552, 1673.9793792949476, 
1255.6737782679425, 1085.8383314421608, 469.37153738116569, 572.62116896081454, 311.77434912238351, 
281.2991158353845, 207.23629472124972, 153.00771230437678, 182.20778697321808, 143.43840468084434, 
118.77765094695116, 696.76759225787612, 516.62377320306996, 226.90904684952216, 158.84854273840233, 
134.58695232421391, 101.4555311931308, 67.405015189180361, 68.261740327194843, 51.614226738044174, 
40.954491518744412, 110.36649619582273, 467.52886314809808, 455.34564097165105, 265.99869526897623, 
199.29353305227377, 131.84961067170869, 111.2715040326446, 71.16662096823201, 38.146152389476846, 
38.6990769027299, 59.870692200788461, 443.4793969370599, 290.88194395286911, 209.09212375009082, 
127.23172931329275, 95.517077846340342, 73.529493847025378, 69.694285620886248, 78.470765178319752, 
56.512638434293081, 295.00801866652381, 306.68800506414971, 265.69831997088272, 167.10696355029174, 
142.49111126528339, 58.550865628481937, 48.287199205321336, 39.672105531330828, 284.78306933647929, 
212.57873743044075, 123.79423776723996, 102.37688240964525, 116.45085482448894, 95.975031594948021, 
35.324120934978659, 274.73068479028979, 216.62974994736817, 162.40968294832649, 63.025604050933651, 
64.302085498822862, 28.016322236551066, 249.3573151285247, 141.67934095457863, 204.28835316894859, 
122.13768840097538, 31.38026263435038, 161.8497450304771, 125.69294403134448, 102.18547185125026, 
15.703130696679326, 115.2711519172633, 163.22270282220006, 14.078571800132206, 181.30715982042483, 21.61097698524501]
"""

