echo "Getting master URL for cluster $1"
out=`/vagrant/spark-1.6.1-bin-hadoop2.6/ec2/spark-ec2-589 get-master $1`

#Get the url for the Spark master.
#Last string in the output array
for word in $out; do
	master=$word
done

echo "Found master at: $master"

#Copy code file to EC2
echo "Copying ec2_run_me.py to master"
scp -i  /home/vagrant/589S2016.pem ec2_run_me.py root@$master:/root/

#Run the code file with Spark
echo "Running ec2_run_me.py on master"
ssh -i  /home/vagrant/589S2016.pem root@$master "/root/spark/bin/spark-submit /root/ec2_run_me.py"