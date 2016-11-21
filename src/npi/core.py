# coding: utf-8

from __future__ import with_statement
from __future__ import absolute_import
import json
from copy import copy

import numpy as np
from io import open

MAX_ARG_NUM = 3
ARG_DEPTH = 10   # 0~9 digit. one-hot.

PG_CONTINUE = 0
PG_RETURN = 1


class IntegerArguments(object):
    depth = ARG_DEPTH
    max_arg_num = MAX_ARG_NUM
    size_of_arguments = depth * max_arg_num

    def __init__(self, args=None, values=None):
        if values is not None:
            self.values = values.reshape((self.max_arg_num, self.depth))
        else:
            self.values = np.zeros((self.max_arg_num, self.depth), dtype=np.float32)

        if args:
            for i, v in enumerate(args):
                self.update_to(i, v)

    def copy(self):
        obj = IntegerArguments()
        obj.values = np.copy(self.values)
        return obj

    def decode_all(self):
        return [self.decode_at(i) for i in xrange(len(self.values))]

    def decode_at(self, index):
        return self.values[index].argmax()

    def update_to(self, index, integer):
        self.values[index] = 0
        self.values[index, int(np.clip(integer, 0, self.depth-1))] = 1

    def __str__(self):
        return "<IA: %s>" % self.decode_all()


class Program(object):
    output_to_env = False

    def __init__(self, name, *args):
        self.name = name
        self.args = args
        self.program_id = None

    def description_with_args(self, args):
        int_args = args.decode_all()
        return "%s(%s)" % (self.name, ", ".join([unicode(x) for x in int_args]))

    def to_one_hot(self, size, dtype=np.float):
        ret = np.zeros((size,), dtype=dtype)
        ret[self.program_id] = 1
        return ret

    def do(self, env, args):
        raise NotImplementedError()

    def __str__(self):
        return "<Program: name=%s>" % self.name


class StepInput(object):
    def __init__(self, env, program, arguments):
        self.env = env
        self.program = program
        self.arguments = arguments


class StepOutput(object):
    def __init__(self, r, program=None, arguments=None):
        self.r = r
        self.program = program
        self.arguments = arguments

    def __str__(self):
        return "<StepOutput: r=%s pg=%s arg=%s>" % (self.r, self.program, self.arguments)


class StepInOut(object):
    def __init__(self, input, output):
        self.input = input
        self.output = output


class ResultLogger(object):
    def __init__(self, filename):
        self.filename = filename

    def write(self, obj):
        with open(self.filename, "a") as f:
            #json.dump(obj, f)
            #f.write('\n')
            pass


class NPIStep(object):
    def reset(self):
        pass

    def enter_function(self):
        pass

    def exit_function(self):
        pass

    def step(self, env_observation, pg, arguments):
        raise NotImplementedError()


class RuntimeSystem(object):
    def __init__(self, terminal=None):
        self.terminal = terminal

    def logging(self, message):
        if self.terminal:
            self.terminal.add_log(message)
        else:
            print message


def to_one_hot_array(idx, size, dtype=np.int8):
    ret = np.zeros((size, ), dtype=dtype)
    ret[idx] = 1
    return ret
