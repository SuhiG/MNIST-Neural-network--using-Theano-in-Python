'''Mini-batch Gradient Descent Algorithm used for MNIST dataset.
no hidden layers
Softmax function used for the output layer

'''
from __future__ import print_function
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import theano as t
import theano.tensor as tt
import timeit


print("Using device", t.config.device)

print("Loading data")

f = gzip.open('mnist.pkl.gz', 'rb') 
train_set, valid_set, test_set = pickle.load(f,encoding="latin1")   #load the mnist dataset (use 'latin1' for python3)
#f.close()

plt.rcParams["figure.figsize"]=(10,10) 
plt.rcParams["image.cmap"]="gray"

#for i in range(9):
  #  plt.subplot(1,10,i+1)
   # plt.imshow(train_set[0][i].reshape(28,28))
   # plt.axis("off")
    #plt.title(str(train_set[1][i]))

#plt.show() 

train_set_x=t.shared(np.asarray(train_set[0],dtype=t.config.floatX))
train_set_y=t.shared(np.asarray(train_set[1],dtype="int32"))

print("Building model")
batch_size=600
n_in=28*28 #inputs
n_out=10    #outputs

x=tt.matrix("x")
y=tt.ivector("y")
w=t.shared(value=np.zeros((n_in,n_out),dtype=t.config.floatX),name="w",borrow=True) #calculating the weights 
b=t.shared(value=np.zeros((n_out,),dtype=t.config.floatX),name="b",borrow=True) #calculating the biases

model=tt.nnet.softmax(tt.dot(x,w)+b) #calculating the output by using matrix dot product with softmax optimization function

y_pred=tt.argmax(model,axis=1) #selecting the maximum value from the predicted outputs
error = tt.mean(tt.neq(y_pred, y))#neq-> a!=b ,when True returns 1 & when false 0
#and 

cost=-tt.mean(tt.log(model)[tt.arange(y.shape[0]),y])

#error=tt.mean(tt.neq(y_pred,y))

g_w=tt.grad(cost=cost,wrt=w)
g_b=tt.grad(cost=cost,wrt=b)

learning_rate=0.13
index=tt.lscalar()

train_model=t.function(
    inputs=[index],
    outputs=[cost,error],
    updates=[(w,w-learning_rate*g_w),(b,b-learning_rate*g_b)],
    givens={
        x:train_set_x[index*batch_size:(index+1)*batch_size],
        y:train_set_y[index*batch_size:(index+1)*batch_size]
    }
)

validate_model = t.function(
    inputs=[x,y],
    outputs=[cost,error]
)

print("Training")

n_epochs=100
#print_every=50000
x_g=[]
y_g=[]
n_train_batches=train_set[0].shape[0]//batch_size

n_iters=n_epochs*n_train_batches
train_loss=np.zeros(n_iters)
train_error=np.zeros(n_iters)

validation_interval = 100
n_valid_batches = valid_set[0].shape[0] // batch_size
valid_loss = np.zeros(n_iters // validation_interval)
valid_error = np.zeros(n_iters // validation_interval)

start_time = timeit.default_timer()

for epoch in range(n_epochs):
    for minibatch_index in range(n_train_batches):
        iteration=minibatch_index+n_train_batches*epoch
        train_loss[iteration],train_error[iteration]=train_model(minibatch_index)

        if iteration % validation_interval == 0 :
            val_iteration = iteration // validation_interval
            valid_loss[val_iteration], valid_error[val_iteration] = np.mean([
                    validate_model(
                        valid_set[0][i * batch_size: (i + 1) * batch_size],
                        np.asarray(valid_set[1][i * batch_size: (i + 1) * batch_size], dtype="int32")
                        )
                        for i in range(n_valid_batches)
                     ],axis=0)
            print('epoch {}, minibatch {}/{}, validation error {:02.2f} %, validation loss {}'.format(
                epoch,
                minibatch_index+1,
                n_train_batches,
                valid_error[val_iteration]*100,
                valid_loss[val_iteration]
            ))
            x_g.append(epoch)
            y_g.append(train_error[iteration]*100)
            
           
end_time = timeit.default_timer()
print("Time duration =", end_time -start_time )


            
print(x_g)            
print(y_g)
plt.plot(x_g,y_g)
plt.show()