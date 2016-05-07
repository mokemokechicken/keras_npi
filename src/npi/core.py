# coding: utf-8

import json
from copy import copy

import numpy as np

MAX_ARG_NUM = 3
ARG_DEPTH = 1

PG_CONTINUE = 0
PG_RETURN = 1


class IntegerArguments:
    depth = ARG_DEPTH
    max_arg_num = MAX_ARG_NUM
    size_of_arguments = depth * max_arg_num

    def __init__(self, args: list=None, values: np.ndarray=None):
        if values is not None:
            self.values = values.reshape((self.max_arg_num, self.depth))
        else:
            self.values = np.zeros((self.max_arg_num, self.depth), dtype=np.float32)
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
        return int(self.values[index])

    def update_to(self, index: int, integer: int):
        self.valid_index.add(index)
        self.values[index] = int(integer)

    def __str__(self):
        return "<IA: %s>" % self.decode_all()


class Program:
    output_to_env = False

    def __init__(self, name, *args):
        self.name = name
        self.args = args
        self.program_id = None

    def description_with_args(self, args: IntegerArguments) -> str:
        int_args = args.decode_all()
        return "%s(%s)" % (self.name, ", ".join([str(x) for x in int_args]))

    def to_one_hot(self, size, dtype=np.float):
        ret = np.zeros((size,), dtype=dtype)
        ret[self.program_id] = 1
        return ret

    def do(self, env, args: IntegerArguments):
        raise NotImplementedError()

    def __str__(self):
        return "<Program: name=%s>" % self.name


class StepInput:
    def __init__(self, env: np.ndarray, program: Program, arguments: IntegerArguments):
        self.env = env
        self.program = program
        self.arguments = arguments


class StepOutput:
    def __init__(self, r: float, program: Program=None, arguments: IntegerArguments=None):
        self.r = r
        self.program = program
        self.arguments = arguments

    def __str__(self):
        return "<StepOutput: r=%s pg=%s arg=%s>" % (self.r, self.program, self.arguments)


class StepInOut:
    def __init__(self, input: StepInput, output: StepOutput):
        self.input = input
        self.output = output


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
    def __init__(self, terminal=None):
        self.terminal = terminal

    def logging(self, message):
        if self.terminal:
            self.terminal.add_log(message)
        else:
            print(message)

