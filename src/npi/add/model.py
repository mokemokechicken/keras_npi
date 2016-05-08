#!/usr/bin/env python
# coding: utf-8
import os

import numpy as np
from keras.engine.topology import Merge, Input, InputLayer
from keras.engine.training import Model
from keras.layers.core import Dense, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_yaml
from keras.regularizers import l1, l2
from keras.utils.visualize_util import plot
import keras.backend as K

from npi.add.config import FIELD_ROW, FIELD_DEPTH, PROGRAM_VEC_SIZE, MAX_PROGRAM_NUM, PROGRAM_KEY_VEC_SIZE, FIELD_WIDTH
from npi.add.lib import AdditionProgramSet, AdditionEnv, run_npi
from npi.core import NPIStep, Program, IntegerArguments, StepOutput, RuntimeSystem, PG_RETURN, StepInOut, StepInput
from npi.terminal_core import TerminalNPIRunner

__author__ = 'k_morishita'


class AdditionNPIModel(NPIStep):
    model = None

    def __init__(self, system: RuntimeSystem, model_path: str=None, program_set: AdditionProgramSet=None):
        self.system = system
        self.model_path = model_path
        self.program_set = program_set
        self.batch_size = 1
        self.build()
        self.load_weights()

    def build(self):
        L1_COST = 0.001
        L2_COST = 0.001
        enc_size = self.size_of_env_observation()
        argument_size = IntegerArguments.size_of_arguments
        input_enc = InputLayer(batch_input_shape=(self.batch_size, enc_size), name='input_enc')
        input_arg = InputLayer(batch_input_shape=(self.batch_size, argument_size), name='input_arg')
        input_prg = Embedding(input_dim=MAX_PROGRAM_NUM, output_dim=PROGRAM_VEC_SIZE, input_length=1,
                              W_regularizer=l1(l=L1_COST),
                              batch_input_shape=(self.batch_size, 1))

        f_enc = Sequential(name='f_enc')
        f_enc.add(Merge([input_enc, input_arg], mode='concat'))
        # f_enc.add(Dense(20, W_regularizer=l1(l=L1_COST)))
        f_enc.add(Reshape((1, enc_size + argument_size)))

        program_embedding = Sequential(name='program_embedding')
        program_embedding.add(input_prg)

        f_lstm = Sequential(name='f_lstm')
        f_lstm.add(Merge([f_enc, program_embedding], mode='concat'))
        # f_lstm.add(Activation('relu', name='relu_lstm_0'))
        # f_lstm.add(LSTM(64, return_sequences=True, stateful=True, W_regularizer=l2(l=L2_COST)))
        # f_lstm.add(Activation('relu', name='relu_lstm_1'))
        f_lstm.add(LSTM(128, return_sequences=False, stateful=True, W_regularizer=l2(l=L2_COST)))
        # f_lstm.add(Activation('relu', name='relu_lstm_2'))
        # plot(f_lstm, to_file='f_lstm.png', show_shapes=True)

        f_end = Sequential(name='f_end')
        f_end.add(f_lstm)
        f_end.add(Dense(10, W_regularizer=l1(l=L1_COST)))
        f_end.add(Dense(1, W_regularizer=l1(l=L1_COST)))
        f_end.add(Activation('sigmoid', name='sigmoid_end'))
        # plot(f_end, to_file='f_end.png', show_shapes=True)

        f_prog = Sequential(name='f_prog')
        f_prog.add(f_lstm)
        f_prog.add(Dense(PROGRAM_KEY_VEC_SIZE, W_regularizer=l1(l=L1_COST)))
        f_prog.add(Dense(PROGRAM_VEC_SIZE, W_regularizer=l1(l=L1_COST)))
        f_prog.add(Activation('softmax', name='softmax_prog'))
        # plot(f_prog, to_file='f_prog.png', show_shapes=True)

        f_arg = Sequential(name='f_arg')
        f_arg.add(f_lstm)
        # f_arg.add(Dense(64, W_regularizer=l1(l=L1_COST*0.01)))
        f_arg.add(Dense(argument_size, W_regularizer=l1(l=L1_COST*0.01)))
        f_arg.add(Activation('relu', name='relu_arg'))
        # plot(f_arg, to_file='f_arg.png', show_shapes=True)

        model = Model([input_enc.input, input_arg.input, input_prg.input],
                      [f_end.output, f_prog.output, f_arg.output],
                      name="npi")
        model.compile(optimizer='rmsprop',
                      loss=['binary_crossentropy', 'categorical_crossentropy', 'mean_squared_error'],
                      loss_weights=[0.25, 0.25, 1.0])
        plot(model, to_file='model.png', show_shapes=True)

        self.model = model

    def reset(self):
        super(AdditionNPIModel, self).reset()
        for l in self.model.layers:
            if type(l) is LSTM:
                l.reset_states()

    def fit(self, steps_list, epoch=100):
        """

        :param int epoch:
        :param typing.List[typing.Dict[q=dict, steps=typing.List[StepInOut]]] steps_list:
        :return:
        """

        addition_env = AdditionEnv(FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH)
        npi_runner = TerminalNPIRunner(None, self)

        for ep in range(1, epoch+1):
            for idx, steps_dict in enumerate(steps_list[:40]):
                question = steps_dict['q']
                if self.question_test(addition_env, npi_runner, question):
                    print("GOOD!: ep=%2d idx=%s :%s" % (ep, idx, question))
                    continue

                steps = steps_dict['steps']
                xs = []
                ys = []
                ws = []
                for step in steps:
                    xs.append(self.convert_input(step.input))
                    y, w = self.convert_output(step.output)
                    ys.append(y)
                    ws.append(w)

                it = -1
                while True:
                    it += 1
                    self.reset()
                    losses = []

                    for i, (x, y, w) in enumerate(zip(xs, ys, ws)):
                        loss = self.model.train_on_batch(x, y, sample_weight=w)
                        losses.append(loss)
                    print("ep=%2d idx=%s %s: ave loss %.3f" % (ep, idx, it, np.average(losses)))

                    if it % 3 == 0 and self.question_test(addition_env, npi_runner, question):
                        break

                if idx % 10 == 0:
                    self.save()
                    print("save model")

    def question_test(self, addition_env, npi_runner, question):
        addition_env.reset()
        self.reset()
        try:
            run_npi(addition_env, npi_runner, self.program_set.ADD, question)
            if question['correct']:
                return True
        except StopIteration:
            pass
        return False

    def convert_input(self, p_in: StepInput):
        x_pg = np.array((p_in.program.program_id,))
        x = [xx.reshape((self.batch_size, -1)) for xx in (p_in.env, p_in.arguments.values, x_pg)]
        return x

    def convert_output(self, p_out: StepOutput):
        y = [np.array((p_out.r,))]
        weights = [[1]]
        if p_out.program:
            y += [p_out.program.to_one_hot(PROGRAM_VEC_SIZE), p_out.arguments.values]
            if p_out.program.args:
                weights += [[1], [1]]
            else:
                weights += [[1], [1e-10]]
        else:
            y += [np.zeros((PROGRAM_VEC_SIZE, )), IntegerArguments().values]
            weights += [[1e-10], [1e-10]]
        weights = [np.array(w) for w in weights]
        return [yy.reshape((self.batch_size, -1)) for yy in y], weights

    def step(self, env_observation: np.ndarray, pg: Program, arguments: IntegerArguments) -> StepOutput:
        x = self.convert_input(StepInput(env_observation, pg, arguments))
        r, pg_one_hot, args_value = self.model.predict(x, batch_size=1)  # if batch_size==1, returns single row
        program = self.program_set.get(pg_one_hot.argmax())
        ret = StepOutput(r, program, IntegerArguments(values=args_value))
        return ret

    def save(self):
        self.model.save_weights(self.model_path, overwrite=True)

    def load_weights(self):
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)

    @staticmethod
    def size_of_env_observation():
        return FIELD_ROW * FIELD_DEPTH
