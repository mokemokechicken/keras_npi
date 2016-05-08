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
from keras.optimizers import Adam
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
        # f_enc.add(Dense(35, W_regularizer=l1(l=L1_COST)))
        f_enc.add(Reshape((1, enc_size + argument_size)))

        program_embedding = Sequential(name='program_embedding')
        program_embedding.add(input_prg)

        f_lstm = Sequential(name='f_lstm')
        f_lstm.add(Merge([f_enc, program_embedding], mode='concat'))
        # f_lstm.add(Activation('relu', name='relu_lstm_0'))
        # f_lstm.add(LSTM(256, return_sequences=True, stateful=True, W_regularizer=l2(l=L2_COST)))
        # f_lstm.add(Activation('relu', name='relu_lstm_1'))
        f_lstm.add(LSTM(256, return_sequences=False, stateful=True, W_regularizer=l2(l=L2_COST)))
        f_lstm.add(Activation('relu', name='relu_lstm_2'))
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

        f_args = []
        for ai in range(1, IntegerArguments.max_arg_num+1):
            f_arg = Sequential(name='f_arg%s' % ai)
            f_arg.add(f_lstm)
            f_arg.add(Dense(32, W_regularizer=l1(l=L1_COST*0.01)))
            f_arg.add(Dense(IntegerArguments.depth, W_regularizer=l1(l=L1_COST*0.01)))
            f_arg.add(Activation('softmax', name='softmax_arg%s' % ai))
            f_args.append(f_arg)
        # plot(f_arg, to_file='f_arg.png', show_shapes=True)

        self.model = Model([input_enc.input, input_arg.input, input_prg.input],
                           [f_end.output, f_prog.output] + [fa.output for fa in f_args],
                           name="npi")
        self.compile_model()
        plot(self.model, to_file='model.png', show_shapes=True)

    def reset(self):
        super(AdditionNPIModel, self).reset()
        for l in self.model.layers:
            if type(l) is LSTM:
                l.reset_states()

    def compile_model(self, lr=0.0001):
        arg_num = IntegerArguments.max_arg_num
        optimizer = Adam(lr=lr)
        loss = ['binary_crossentropy', 'categorical_crossentropy'] + ['categorical_crossentropy'] * arg_num
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=[0.25, 0.25] + [1] * arg_num)

    def fit(self, steps_list, epoch=3000):
        """

        :param int epoch:
        :param typing.List[typing.Dict[q=dict, steps=typing.List[StepInOut]]] steps_list:
        :return:
        """

        def filter_question(condition_func):
            sub_steps_list = []
            for steps_dict in steps_list:
                question = steps_dict['q']
                if condition_func(question['in1'], question['in2']):
                    sub_steps_list.append(steps_dict)
            return sub_steps_list

        q_type = "training questions of a+b < 10"
        print(q_type)
        all_ok = self.fit_to_subset(filter_question(lambda a, b: a+b < 10), epoch=epoch)
        print("%s is all_ok=%s" % (q_type, all_ok))

        q_type = "training questions of 10 <= a+b < 20"
        print(q_type)
        all_ok = self.fit_to_subset(filter_question(lambda a, b: 10 <= a + b < 20), epoch=epoch)
        print("%s is all_ok=%s" % (q_type, all_ok))

        q_type = "training questions of a<10 and b<10"
        print(q_type)
        all_ok = self.fit_to_subset(filter_question(lambda a, b: a < 10 and b < 10), epoch=epoch)
        print("%s is all_ok=%s" % (q_type, all_ok))

        q_type = "training questions of a<100 and b<100"
        print(q_type)
        all_ok = self.fit_to_subset(filter_question(lambda a, b: a < 100 and b < 100), epoch=epoch)
        print("%s is all_ok=%s" % (q_type, all_ok))

        q_type = "training questions of ALL"
        print(q_type)
        all_ok = self.fit_to_subset(filter_question(lambda a, b: True), epoch=epoch)
        print("%s is all_ok=%s" % (q_type, all_ok))

    def fit_to_subset(self, steps_list, epoch=3000):
        learning_rate = 0.0001
        print("Re-Compile Model lr=%s" % learning_rate)
        self.compile_model(learning_rate)

        addition_env = AdditionEnv(FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH)
        npi_runner = TerminalNPIRunner(None, self)
        all_ok = False
        for ep in range(1, epoch+1):
            if all_ok:
                break
            all_ok = True
            losses = []
            ok_rate = []
            np.random.shuffle(steps_list)
            for idx, steps_dict in enumerate(steps_list):
                question = steps_dict['q']
                if self.question_test(addition_env, npi_runner, question):
                    print("GOOD!: ep=%2d idx=%s :%s" % (ep, idx, question))
                    ok_rate.append(1)
                    continue
                ok_rate.append(0)
                all_ok = False

                steps = steps_dict['steps']
                xs = []
                ys = []
                ws = []
                for step in steps:
                    xs.append(self.convert_input(step.input))
                    y, w = self.convert_output(step.output)
                    ys.append(y)
                    ws.append(w)

                self.reset()

                for i, (x, y, w) in enumerate(zip(xs, ys, ws)):
                    loss = self.model.train_on_batch(x, y, sample_weight=w)
                    losses.append(loss)
            if losses:
                print("ep=%2d: ok_rate=%.2f%% ave loss %.3f (%s samples)" % (ep, np.average(ok_rate)*100, np.average(losses), len(steps_list)))
            self.save()

            if ep % 50 == 0:
                learning_rate *= 0.95
                print("Re-Compile Model lr=%s" % learning_rate)
                self.compile_model(learning_rate)
        return all_ok

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
        weights = [[1.]]
        if p_out.program:
            arg_values = p_out.arguments.values
            arg_num = len(p_out.program.args or [])
            y += [p_out.program.to_one_hot(PROGRAM_VEC_SIZE)]
            weights += [[1.]]
        else:
            arg_values = IntegerArguments().values
            arg_num = 0
            y += [np.zeros((PROGRAM_VEC_SIZE, ))]
            weights += [[1e-10]]

        for v in arg_values:  # split by each args
            y += [v]
        weights += [[1.]] * arg_num + [[1e-10]] * (len(arg_values) - arg_num)
        weights = [np.array(w) for w in weights]
        return [yy.reshape((self.batch_size, -1)) for yy in y], weights

    def step(self, env_observation: np.ndarray, pg: Program, arguments: IntegerArguments) -> StepOutput:
        x = self.convert_input(StepInput(env_observation, pg, arguments))
        results = self.model.predict(x, batch_size=1)  # if batch_size==1, returns single row

        r, pg_one_hot, arg_values = results[0], results[1], results[2:]
        program = self.program_set.get(pg_one_hot.argmax())
        ret = StepOutput(r, program, IntegerArguments(values=np.stack(arg_values)))
        return ret

    def save(self):
        self.model.save_weights(self.model_path, overwrite=True)

    def load_weights(self):
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)

    @staticmethod
    def size_of_env_observation():
        return FIELD_ROW * FIELD_DEPTH
