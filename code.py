#sc = spark.sparkContext
import pickle
import os
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

rel_test = "pokemon\test_batch.pickle"
rel_train1 = "pokemon\train_batch_1.pickle"
rel_train2 = "pokemon\train_batch_2.pickle"
rel_train3 = "pokemon\train_batch_3.pickle"
rel_train4 = "pokemon\train_batch_4.pickle"
rel_train5 = "pokemon\train_batch_5.pickle"

#test = os.path.join(script_dir, rel_test)
#train1 = os.path.join(script_dir, rel_train1)
#train2 = os.path.join(script_dir, rel_train2)
#train3 = os.path.join(script_dir, rel_train3)
#train4 = os.path.join(script_dir, rel_train4)
#train5 = os.path.join(script_dir, rel_train5)

test = "pokemon\test_batch.pickle"
train1 = "pokemon\train_batch_1.pickle"
train2 = "pokemon\train_batch_2.pickle"
train3 = "pokemon\train_batch_3.pickle"
train4 = "pokemon\train_batch_4.pickle"
train5 = "pokemon\train_batch_5.pickle" 

infile = open(test,'rb')
test_dict = pickle.load(infile)
infile.close()
print(test_dict)

#C:\Users\csvin\OneDrive\Desktop\BD_Project\pokemon\code.py

#df = spark.read.load("examples/src/main/resources/people.csv", format="csv", sep=":", inferSchema="true", header="true")