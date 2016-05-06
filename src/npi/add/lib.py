# coding: utf-8

import numpy as np

from npi.terminal_core import Screen, Terminal
from npi.core import Program, IntegerArguments, StepResult, NPIStep

__author__ = 'k_morishita'


PG_CONTINUE = 0
PG_RETURN = 1


class AdditionEnv:
    """
    Environment of Addition
    """
    def __init__(self, height, width, num_chars):
        self.screen = Screen(height, width)
        self.num_chars = num_chars
        self.pointers = [0] * height
        self.reset()

    def reset(self):
        self.screen.fill(0)
        self.pointers = [self.screen.width-1] * self.screen.height  # rightmost

    def get_observation(self) -> np.ndarray:
        value = []
        for row in range(len(self.pointers)):
            value.append(self.to_one_hot(self.screen[row, self.pointers[row]]))
        return np.array(value)  # shape of FIELD_ROW * FIELD_DEPTH

    def to_one_hot(self, ch):
        ret = np.zeros((self.num_chars,), dtype=np.int8)
        if 0 <= ch < self.num_chars:
            ret[ch] = 1
        else:
            raise IndexError("ch must be 0 <= ch < %s, but %s" % (self.num_chars, ch))
        return ret

    def setup_problem(self, num1, num2):
        for i, s in enumerate(reversed("%s" % num1)):
            self.screen[0, -(i+1)] = int(s) + 1
        for i, s in enumerate(reversed("%s" % num2)):
            self.screen[1, -(i+1)] = int(s) + 1

    def move_pointer(self, row, left_or_right):
        self.pointers[row] += 1 if left_or_right == 1 else -1  # LEFT is 0, RIGHT is 1
        self.pointers[row] %= self.screen.width

    def write(self, row, ch):
        row %= self.screen.height
        self.screen[row, self.pointers[row]] = ch

    def get_output(self):
        s = ""
        for ch in self.screen[3]:
            if ch > 0:
                s += "%s" % (ch-1)
        return int(s or "0")


class MovePtrProgram(Program):
    output_to_env = True
    PTR_IN1 = 0
    PTR_IN2 = 1
    PTR_CARRY = 2
    PTR_OUT = 3

    TO_LEFT = 0
    TO_RIGHT = 1

    def do(self, env: AdditionEnv, args: IntegerArguments):
        ptr_kind = args.decode_at(0)
        left_or_right = args.decode_at(1)
        env.move_pointer(ptr_kind, left_or_right)


class WriteProgram(Program):
    output_to_env = True
    WRITE_TO_CARRY = 0
    WRITE_TO_OUTPUT = 1

    def do(self, env: AdditionEnv, args: IntegerArguments):
        row = 2 if args.decode_at(0) == self.WRITE_TO_CARRY else 3
        digit = args.decode_at(1)
        env.write(row, digit+1)


class AdditionProgramSet:
    MOVE_PTR = MovePtrProgram('MOVE_PTR', 4, 2)  # PTR_KIND(4), LEFT_OR_RIGHT(2)
    WRITE = WriteProgram('WRITE', 2, 10)       # CARRY_OR_OUT(2), DIGITS(10)
    ADD = Program('ADD')
    ADD1 = Program('ADD1')
    CARRY = Program('CARRY')
    LSHIFT = Program('LSHIFT')
    RSHIFT = Program('RSHIFT')

    def __init__(self):
        self.map = {}
        self.program_id = 0
        self.register(self.MOVE_PTR)
        self.register(self.WRITE)
        self.register(self.ADD)
        self.register(self.ADD1)
        self.register(self.CARRY)
        self.register(self.LSHIFT)
        self.register(self.RSHIFT)

    def register(self, pg: Program):
        pg.program_id = self.program_id
        self.map[pg.program_id] = pg
        self.program_id += 1

    def get(self, i: int):
        return self.map.get(i)


class AdditionTeacher(NPIStep):
    def __init__(self, program_set: AdditionProgramSet, terminal: Terminal):
        self.pg_set = program_set
        self.terminal = terminal  # as logger
        self.step_queue = None
        self.step_queue_stack = []
        self.sub_program = {}
        self.register_subprogram(program_set.MOVE_PTR, self.pg_primitive)
        self.register_subprogram(program_set.WRITE   , self.pg_primitive)
        self.register_subprogram(program_set.ADD     , self.pg_add)
        self.register_subprogram(program_set.ADD1    , self.pg_add1)
        self.register_subprogram(program_set.CARRY   , self.pg_carry)
        self.register_subprogram(program_set.LSHIFT  , self.pg_lshift)
        self.register_subprogram(program_set.RSHIFT  , self.pg_rshift)

    def reset(self):
        self.step_queue_stack = []
        self.step_queue = None

    def register_subprogram(self, pg, method):
        self.sub_program[pg.program_id] = method

    @staticmethod
    def decode_params(env_observation: np.ndarray, arguments: IntegerArguments):
        return env_observation.argmax(axis=1), arguments.decode_all()

    def enter_function(self):
        self.step_queue_stack.append(self.step_queue or [])
        self.step_queue = None

    def exit_function(self):
        self.step_queue = self.step_queue_stack.pop()

    def step(self, env_observation: np.ndarray, pg: Program, arguments: IntegerArguments) -> StepResult:
        if not self.step_queue:
            self.step_queue = self.sub_program[pg.program_id](env_observation, arguments)
        if self.step_queue:
            ret = self.convert_for_step_return(self.step_queue[0])
            self.step_queue = self.step_queue[1:]
        else:
            ret = StepResult(PG_RETURN, None, None)
        return ret

    @staticmethod
    def convert_for_step_return(step_values: tuple) -> StepResult:
        if len(step_values) == 2:
            return StepResult(PG_CONTINUE, step_values[0], IntegerArguments(step_values[1]))
        else:
            return StepResult(step_values[0], step_values[1], IntegerArguments(step_values[2]))

    @staticmethod
    def pg_primitive(env_observation: np.ndarray, arguments: IntegerArguments):
        return None

    def pg_add(self, env_observation: np.ndarray, arguments: IntegerArguments):
        ret = []
        (in1, in2, carry, output), (a1, a2, a3) = self.decode_params(env_observation, arguments)
        if in1 == 0 and in2 == 0 and carry == 0:
            return None
        ret.append((self.pg_set.ADD1, None))
        ret.append((self.pg_set.LSHIFT, None))
        return ret

    def pg_add1(self, env_observation: np.ndarray, arguments: IntegerArguments):
        ret = []
        p = self.pg_set
        (in1, in2, carry, output), (a1, a2, a3) = self.decode_params(env_observation, arguments)
        result = self.sum_ch_list([in1, in2, carry])
        ret.append((p.WRITE, (p.WRITE.WRITE_TO_OUTPUT, result % 10)))
        if result > 9:
            ret.append((p.CARRY, None))
        ret[-1] = (PG_RETURN, ret[-1][0], ret[-1][1])
        return ret

    @staticmethod
    def sum_ch_list(ch_list):
        ret = 0
        for ch in ch_list:
            if ch > 0:
                ret += ch - 1
        return ret

    def pg_carry(self, env_observation: np.ndarray, arguments: IntegerArguments):
        ret = []
        p = self.pg_set
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_CARRY, p.MOVE_PTR.TO_LEFT)))
        ret.append((p.WRITE, (p.WRITE.WRITE_TO_CARRY, 1)))
        ret.append((PG_RETURN, p.MOVE_PTR, (p.MOVE_PTR.PTR_CARRY, p.MOVE_PTR.TO_RIGHT)))
        return ret

    def pg_lshift(self, env_observation: np.ndarray, arguments: IntegerArguments):
        ret = []
        p = self.pg_set
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_IN1, p.MOVE_PTR.TO_LEFT)))
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_IN2, p.MOVE_PTR.TO_LEFT)))
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_CARRY, p.MOVE_PTR.TO_LEFT)))
        ret.append((PG_RETURN, p.MOVE_PTR, (p.MOVE_PTR.PTR_OUT, p.MOVE_PTR.TO_LEFT)))
        return ret

    def pg_rshift(self, env_observation: np.ndarray, arguments: IntegerArguments):
        ret = []
        p = self.pg_set
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_IN1, p.MOVE_PTR.TO_RIGHT)))
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_IN2, p.MOVE_PTR.TO_RIGHT)))
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_CARRY, p.MOVE_PTR.TO_RIGHT)))
        ret.append((PG_RETURN, p.MOVE_PTR, (p.MOVE_PTR.PTR_OUT, p.MOVE_PTR.TO_RIGHT)))
        return ret


class AdditionNPIModel(NPIStep):
    def step(self, env_observation: np.ndarray, pg: Program, arguments: IntegerArguments) -> StepResult:
        return StepResult(PG_RETURN, None, None)

    def fit(self, step_list):
        pass
