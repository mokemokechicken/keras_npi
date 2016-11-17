About
=====

Implementation of [Neural Programmer-Interpreters](http://arxiv.org/abs/1511.06279) with Keras.

How to Demo
===========

[Demo Movie](https://youtu.be/s7PuBqwI2YA)

requirement
-----------

* Python3

setup
-----

```
pip install -r requirements.txt
```

create training dataset
-----------------------
### create training dataset
```
sh src/run_create_addition_data.sh
```

### create training dataset with showing steps on terminal
```
DEBUG=1 sh src/run_create_addition_data.sh
```

training model
------------------
### Create New Model (-> remove old model if exists and then create new model)
```
NEW_MODEL=1 sh src/run_train_addition_model.sh
```

### Training Existing Model (-> if a model exists, use the model)
```
sh src/run_train_addition_model.sh
```

test model
----------
### check the model accuracy
```
sh src/run_test_addition_model.sh
```

### check the model accuracy with showing steps on terminal
```
DEBUG=1 sh src/run_test_addition_model.sh
```

Implementation FAQ
==================
These are questions about implementation that I received in the past.

about pydot
-----------
Q: I am using Python3. I am getting an error "module 'pydot' has no attribute 'find_graphviz'".

A: Let's try `pydot-ng`. 
 
`train_f_enc` method
--------------------
Q: What is the purpose of 'env_model' in 'train_f_enc' method which gets called by 'fit' method? My guess is, it is to train the weights of 'f_enc' layer.

A: Yes, that's right.

----

Q: Why is the target output of 'env_model' - [[first digit of sum], [ carry of sum]]? 
Also, why does the target output not have 'output'
As per my understanding the weights of 'f_enc' layer should be trained only in 'self.model'.

A: Yes, in the original paper, 'f_enc' is trained with other layers. It is better not to be trained separately.

The reason of that in my implementation is just difficulty to train the model.
Especially it seemed to hard to train layers before LSTMs (like f_enc layer). 
f_enc weights often became some NaNs. (I don't know why... keras problem? or ??)
So, I tried to train f_enc separately, and it seemed good (not best).

NOP program
-----------
Q: what's the purpose of NOP program?

A: I do not remember it much, but NOP (No Operation) is program_id = 0.
I thought that in the early days of learning, the predicted value often becomes 0, and harmless NOPs that do not perform unnecessary movements will learn more efficiently.
Although it is not certain whether it is effective...

`weights = [[1.]]`
------------------

Q: what's the purpose of `weights = [[1.]]` this initialization?

A: You mean `weights = [[1.]]` in `AdditionNPIModel#convert_output()`, don't you?

The `weights` means learning weights of [f_end, f_prog, f_args].
The first `weights = [[1.]]` means "f_end's weight=1".
f_prog and f_args weights are set to 1 if the teacher returns valid values.