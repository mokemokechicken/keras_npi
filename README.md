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
