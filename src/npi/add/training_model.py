# coding: utf-8
import os
import pickle

from npi.add.config import FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH
from npi.add.lib import AdditionEnv, AdditionProgramSet, AdditionTeacher, create_char_map, create_questions, run_npi
from npi.add.model import AdditionNPIModel
from npi.core import ResultLogger, RuntimeSystem
from npi.terminal_core import TerminalNPIRunner, Terminal


def main(filename: str, model_path: str):
    system = RuntimeSystem()
    program_set = AdditionProgramSet()

    with open(filename, 'rb') as f:
        steps_list = pickle.load(f)

    npi_model = AdditionNPIModel(system, model_path, program_set)
    npi_model.fit(steps_list)


if __name__ == '__main__':
    import sys
    DEBUG_MODE = os.environ.get('DEBUG')
    train_filename = sys.argv[1]
    model_output = sys.argv[2]
    main(train_filename, model_output)

