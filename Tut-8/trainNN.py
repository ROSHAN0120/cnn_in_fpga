#import split_data
import network2
import mnist_loader

# Assume 'data' is your dataset
# data = cyclone_loader.load_data()
#dataset = data[0]
# data_list = list(data)


training_data,validation_data,test_data = mnist_loader.load_data_wrapper()
net = network2.Network([102400,30,30,128])#change the number of layers or number of neurons in each layer here
validation_data = list(test_data)
training_data = list(training_data)
print(training_data[90])

net.SGD(training_data, 50,10,10, lmbda=5.0,evaluation_data=validation_data, monitor_evaluation_accuracy=True)
net.save("Weits&BiasesCyclone.txt") 