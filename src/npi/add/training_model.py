# coding: utf-8
from __future__ import with_statement
from __future__ import absolute_import
import os
import pickle

from npi.add.config import FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH
from npi.add.lib import AdditionEnv, AdditionProgramSet, AdditionTeacher, create_char_map, create_questions, run_npi
from npi.add.model import AdditionNPIModel
from npi.core import ResultLogger, RuntimeSystem
from npi.terminal_core import TerminalNPIRunner, Terminal
from io import open


def main(filename, model_path):
    system = RuntimeSystem()
    program_set = AdditionProgramSet()

    with open(filename, u'rb') as f:
        steps_list = pickle.load(f)

    npi_model = AdditionNPIModel(system, model_path, program_set)
    npi_model.fit(steps_list)


if __name__ == u'__main__':
    import sys
    DEBUG_MODE = os.environ.get(u'DEBUG')
    train_filename = sys.argv[1]
    model_output = sys.argv[2]
    main(train_filename, model_output)

