# coding: utf-8

import json
from collections import namedtuple
from copy import copy

import numpy as np

__author__ = 'k_morishita'

MAX_ARG_NUM = 3    #
ARG_DEPTH = 8  # 8bit integer


StepInput = namedtuple('StepInput',  ['env', 'program', 'arguments'])
StepOutput = namedtuple('StepOutput', ['r', 'program', 'arguments'])
StepInOut = namedtuple('StepInOut', ['input', 'output'])


class IntegerArguments:
    depth = ARG_DEPTH
    max_arg_num = MAX_ARG_NUM
    size_of_arguments = depth * max_arg_num

    def __init__(self, args: list=None):
        self.values = np.zeros((self.max_arg_num, self.depth), dtype=np.int8)
        self.valid_index = set()

        if args:
            for i, v in enumerate(args):
                self.update_to(i, v)

    def copy(self):
        obj = IntegerArguments()
        obj.values = np.copy(self.values)
        obj.valid_index = copy(self.valid_index)
        return obj

    def decode_all(self):
        return [self.decode_at(i) for i in range(len(self.values))]

    def decode_at(self, index: int):
        arg = self.values[index]
        return sum([x*(2**i) for i, x in enumerate(arg)])

    def update_to(self, index: int, integer: int):
        self.valid_index.add(index)
        arg = self.values[index]
        n = int(integer)
        for i in range(len(arg)):
            arg[i] = n % 2
            n //= 2


class Program:
    output_to_env = False

    def __init__(self, name, *args):
        self.name = name
        self.args = args
        self.program_id = None

    def description_with_args(self, args: IntegerArguments) -> str:
        int_args = args.decode_all()
        return "%s(%s)" % (self.name, ", ".join([str(x) for x in int_args]))

    def do(self, env, args: IntegerArguments):
        raise NotImplementedError()

    def __str__(self):
        return "<Program: name=%s>" % self.name


class ResultLogger:
    def __init__(self, filename):
        self.filename = filename

    def write(self, obj):
        with open(self.filename, "a") as f:
            json.dump(obj, f)
            f.write("\n")


class NPIStep:
    def reset(self):
        pass

    def enter_function(self):
        pass

    def exit_function(self):
        pass

    def step(self, env_observation: np.ndarray, pg: Program, arguments: IntegerArguments) -> StepOutput:
        raise NotImplementedError()


class RuntimeSystem:
    def __init__(self):
        pass

    def logging(self, message):
        print(message)


PG_CONTINUE = 0
PG_RETURN = 1