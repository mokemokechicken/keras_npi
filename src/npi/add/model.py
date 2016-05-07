#!/usr/bin/env python
# coding: utf-8
import numpy as np
from keras.engine.topology import Merge, Input, InputLayer
from keras.engine.training import Model
from keras.layers.core import Dense, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.visualize_util import plot
import keras.backend as K

from npi.add.config import FIELD_ROW, FIELD_DEPTH, PROGRAM_VEC_SIZE, MAX_PROGRAM_NUM, PROGRAM_KEY_VEC_SIZE
from npi.core import NPIStep, Program, IntegerArguments, StepOutput, RuntimeSystem, PG_RETURN, StepInOut

__author__ = 'k_morishita'


class AdditionNPIModel(NPIStep):
    model = None

    def __init__(self, system: RuntimeSystem):
        self.system = system
        self.batch_size = 1
        self.build()

    def build(self):
        input_enc = InputLayer(batch_input_shape=(self.batch_size, self.size_of_env_observation()), name='input_enc')
        input_arg = InputLayer(batch_input_shape=(self.batch_size, IntegerArguments.size_of_arguments), name='input_arg')
        input_prg = Embedding(input_dim=MAX_PROGRAM_NUM, output_dim=PROGRAM_VEC_SIZE, input_length=1,
                              batch_input_shape=(self.batch_size, 1))

        f_enc = Sequential(name='f_enc')
        f_enc.add(Merge([input_enc, input_arg], mode='concat'))
        f_enc.add(Dense(20))
        f_enc.add(Reshape((1, 20)))

        program_embedding = Sequential(name='program_embedding')
        program_embedding.add(input_prg)

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

        model = Model([*f_enc.inputs, program_embedding.input], [f_end.output, f_prog.output, f_arg.output], name="npi")
        model.compile(optimizer='rmsprop', loss=['binary_crossentropy', 'categorical_crossentropy', 'mean_squared_error'])
        plot(model, to_file='model.png', show_shapes=True)

        self.model = model

    def reset_lstm(self):
        for l in self.model.layers:
            if type(l) is LSTM:
                l.reset_states()

    def fit(self, steps_list):
        """

        :param typing.List[typing.List[StepInOut]] steps_list:
        :return:
        """
        for steps in steps_list:
            self.reset_lstm()
            xs = []
            ys = []
            for step in steps:
                # INPUT
                i = step.input
                x_pg = np.array((i.program.program_id,))
                x = [xx.reshape((1, -1)) for xx in (i.env, i.arguments.values, x_pg)]
                xs.append(x)
                # OUTPUT
                o = step.output
                y = [np.array((o.r, ))]
                if o.program:
                    y += [o.program.to_one_hot(PROGRAM_VEC_SIZE), o.arguments.values]
                else:
                    y += [np.zeros((PROGRAM_VEC_SIZE, )), IntegerArguments().values]
                y = [yy.reshape((1, -1)) for yy in y]
                ys.append(y)
            for x, y in zip(xs, ys):
                losses = self.model.train_on_batch(x, y)
                print(losses)

    def step(self, env_observation: np.ndarray, pg: Program, arguments: IntegerArguments) -> StepOutput:
        return StepOutput(PG_RETURN, None, None)

    @staticmethod
    def size_of_env_observation():
        return FIELD_ROW * FIELD_DEPTH
