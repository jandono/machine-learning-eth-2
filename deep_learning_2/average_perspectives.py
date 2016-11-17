import numpy as np

DATA_DIR = "/Users/benjamin/Documents/ETH/machine_learning/project1/transferred_results/"
FILES = ["axis0_correct/test_outputs/outputs_epoch_99.csv", 
	#"axis1_correct/test_outputs/outputs_epoch_99.csv",
	"axis2_correct/test_outputs/outputs_epoch_99.csv"]

aligned = np.zeros(shape=(138,len(FILES)))

for row,file in enumerate(FILES):
	f = open(DATA_DIR + file,'r')
	lines = f.readlines()[1:]
	for i,line in enumerate(lines):
		values = line.split(',')
		aligned[i,row] = values[1]

	f.close()

average = np.mean(aligned,axis=1)

output_file = open(DATA_DIR + "combinations/averages_0_and_2_epoch99_doubles.csv", 'w')
output_file.write("ID,Prediction\n")
for i in range(average.shape[0]):
    output_file.write("{},{} \n".format(i+1,average[i]))