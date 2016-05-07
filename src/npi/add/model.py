#!/usr/bin/env python
# coding: utf-8
import numpy as np
from keras.engine.topology import Merge
from keras.engine.training import Model
from keras.layers.core import Dense, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.visualize_util import plot

from npi.add.config import FIELD_ROW, FIELD_DEPTH, PROGRAM_VEC_SIZE, MAX_PROGRAM_NUM, PROGRAM_KEY_VEC_SIZE
from npi.core import NPIStep, Program, IntegerArguments, StepOutput, RuntimeSystem, PG_RETURN


__author__ = 'k_morishita'


class AdditionNPIModel(NPIStep):
    def __init__(self, system: RuntimeSystem):
        self.system = system
        self.batch_size = 1
        model = self.build()

    def build(self):
        size_of_enc_input = self.size_of_env_observation() + IntegerArguments.size_of_arguments
        f_enc = Sequential(name='f_enc')
        f_enc.add(Dense(20, batch_input_shape=(self.batch_size, size_of_enc_input)))
        f_enc.add(Reshape((1, 20)))

        program_embedding = Sequential(name='program_embedding')
        program_embedding.add(Embedding(input_dim=MAX_PROGRAM_NUM, output_dim=PROGRAM_VEC_SIZE, input_length=1,
                                        batch_input_shape=(self.batch_size, 1)))

        f_lstm = Sequential(name='f_lstm')
        f_lstm.add(Merge([f_enc, program_embedding], mode='concat'))
        f_lstm.add(Activation('relu', name='relu_0'))
        f_lstm.add(LSTM(256, return_sequences=True, stateful=True))
        f_lstm.add(Activation('relu', name='relu_1'))
        f_lstm.add(LSTM(256, return_sequences=False, stateful=True))
        f_lstm.add(Activation('relu', name='relu_2'))
        # plot(f_lstm, to_file='f_lstm.png', show_shapes=True)

        f_end = Sequential(name='f_end')
        f_end.add(f_lstm)
        f_end.add(Dense(1))
        f_end.add(Activation('sigmoid', name='sigmoid_1'))
        # plot(f_end, to_file='f_end.png', show_shapes=True)

        f_prog = Sequential(name='f_prog')
        f_prog.add(f_lstm)
        f_prog.add(Dense(PROGRAM_KEY_VEC_SIZE))
        f_prog.add(Dense(PROGRAM_VEC_SIZE))
        f_prog.add(Activation('sigmoid', name='sigmoid_2'))
        # plot(f_prog, to_file='f_prog.png', show_shapes=True)

        f_arg = Sequential(name='f_arg')
        f_arg.add(f_lstm)
        f_arg.add(Dense(IntegerArguments.size_of_arguments))
        f_arg.add(Activation('sigmoid', name='sigmoid_3'))
        # plot(f_arg, to_file='f_arg.png', show_shapes=True)

        model = Model([f_enc.input, program_embedding.input], [f_end.output, f_prog.output, f_arg.output], name="npi")
        plot(model, to_file='model.png', show_shapes=True)


        return model

    def fit(self, step_list):
        pass

    def step(self, env_observation: np.ndarray, pg: Program, arguments: IntegerArguments) -> StepOutput:
        return StepOutput(PG_RETURN, None, None)

    @staticmethod
    def size_of_env_observation():
        return FIELD_ROW * FIELD_DEPTH
