# coding: utf-8
import curses
from random import random

from npi.core import IntegerArguments, ResultLogger
from npi.add.lib import AdditionEnv, AdditionProgramSet, AdditionTeacher, AdditionNPIModel
from npi.terminal_core import TerminalNPIRunner, Terminal, show_env_to_terminal

FIELD_ROW = 4     # Input1, Input2, Carry, Output
FIELD_WIDTH = 9   # number of columns
FIELD_DEPTH = 11  # number of characters(0~9 digits) and white space, per cell. one-hot-encoding

PROGRAM_VEC_SIZE = 10
KEY_VEC_SIZE = 5


def create_char_map():
    char_map = dict((i+1, "%s" % i) for i in range(10))
    char_map[0] = ' '
    return char_map


def main(stdscr, result_logger: ResultLogger):
    terminal = Terminal(stdscr, create_char_map())
    terminal.init_window(FIELD_WIDTH, FIELD_ROW)
    program_set = AdditionProgramSet()
    addition_env = AdditionEnv(FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH)

    questions = create_questions()
    teacher = AdditionTeacher(program_set, terminal)
    npi_runner = TerminalNPIRunner(terminal, teacher)
    npi_runner.verbose = len(questions) < 10
    for data in questions:
        addition_env.reset()
        run_npi(addition_env, npi_runner, program_set.ADD, data)
        result_logger.write(data)
        terminal.add_log(data)

    npi_model = AdditionNPIModel()
    npi_model.fit(npi_runner.step_list)

    npi_runner.model = npi_model
    for data in create_questions():
        addition_env.reset()
        run_npi(addition_env, npi_runner, program_set.ADD, data)
        result_logger.write(data)
        terminal.add_log(data)


def create_questions(num=100):
    questions = [
        dict(in1=1, in2=4),
        dict(in1=30, in2=50),
        dict(in1=36, in2=85),
        dict(in1=104, in2=902),
    ]

    for _ in range(num):
        questions.append(dict(in1=int(random() * 10000), in2=int(random() * 10000)))
    return questions

# input_params = np.concatenate((env.encoded_value().reshape(-1), args.reshape(-1)))
# f_enc = Sequential()
# f_enc.add(Dense(20, input_dim=input_params.shape[0]))


def run_npi(addition_env, npi_runner, program, data):
    data['expect'] = data['in1'] + data['in2']

    addition_env.setup_problem(data['in1'], data['in2'])

    npi_runner.reset()
    npi_runner.npi_program_interface(addition_env, program, IntegerArguments())

    data['result'] = addition_env.get_output()
    data['correct'] = data['result'] == data['expect']


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'result.log'
    curses.wrapper(main, ResultLogger(filename))
