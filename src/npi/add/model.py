#!/usr/bin/env python
# coding: utf-8
import numpy as np

from npi.add.lib import PG_RETURN, AdditionProgramSet, AdditionEnv
from npi.core import NPIStep, Program, IntegerArguments, StepOutput
from npi.terminal_core import Terminal

__author__ = 'k_morishita'


class AdditionNPIModel(NPIStep):
    def __init__(self, program_set: AdditionProgramSet, terminal: Terminal, env: AdditionEnv):
        pass

    def build(self):
        pass

    def fit(self, step_list):
        pass

    def step(self, env_observation: np.ndarray, pg: Program, arguments: IntegerArguments) -> StepOutput:
        return StepOutput(PG_RETURN, None, None)


# input_params = np.concatenate((env.encoded_value().reshape(-1), args.reshape(-1)))
# f_enc = Sequential()
# f_enc.add(Dense(20, input_dim=input_params.shape[0]))
