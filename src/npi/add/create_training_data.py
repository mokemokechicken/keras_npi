# coding: utf-8
import os
import curses
import pickle
from copy import copy

from npi.add.config import FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH
from npi.add.lib import AdditionEnv, AdditionProgramSet, AdditionTeacher, create_char_map, create_questions, run_npi
from npi.core import ResultLogger
from npi.terminal_core import TerminalNPIRunner, Terminal


def main(stdscr, filename: str, num: int, result_logger: ResultLogger):
    terminal = Terminal(stdscr, create_char_map())
    terminal.init_window(FIELD_WIDTH, FIELD_ROW)
    program_set = AdditionProgramSet()
    addition_env = AdditionEnv(FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH)

    questions = create_questions(num)
    teacher = AdditionTeacher(program_set)
    npi_runner = TerminalNPIRunner(terminal, teacher)
    npi_runner.verbose = DEBUG_MODE
    steps_list = []
    for data in questions:
        addition_env.reset()
        q = copy(data)
        run_npi(addition_env, npi_runner, program_set.ADD, data)
        steps_list.append({"q": q, "steps": npi_runner.step_list})
        result_logger.write(data)
        terminal.add_log(data)

    if filename:
        with open(filename, 'wb') as f:
            pickle.dump(steps_list, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    import sys
    DEBUG_MODE = os.environ.get('DEBUG')
    if DEBUG_MODE:
        output_filename = None
        num_data = 3
        log_filename = 'result.log'
    else:
        output_filename = sys.argv[1] if len(sys.argv) > 1 else None
        num_data = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        log_filename = sys.argv[3] if len(sys.argv) > 3 else 'result.log'
    curses.wrapper(main, output_filename, num_data, ResultLogger(log_filename))
    print("create %d training data" % num_data)
