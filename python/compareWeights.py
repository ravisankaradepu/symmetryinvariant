import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

net1 = caffe.Net('examples/mnist/lenet_train_test.prototxt','examples/mnist/lenet1_iter_1.caffemodel',caffe.TEST)
net1.params['conv1'][0].data

net2 = caffe.Net('examples/mnist/lenet_train_test.prototxt','examples/mnist/lenet22_iter_1.caffemodel',caffe.TEST)
net1.params['conv1'][0].data == net22.params['conv1'][0].data





