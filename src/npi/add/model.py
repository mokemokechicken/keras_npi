#!/usr/bin/env python
# coding: utf-8
import os
from collections import Counter
from copy import copy

import math
import numpy as np
from keras.engine.topology import Merge, InputLayer
from keras.engine.training import Model
from keras.layers.core import Dense, Activation, RepeatVector, MaxoutDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_yaml
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.utils.visualize_util import plot

from npi.add.config import FIELD_ROW, FIELD_DEPTH, PROGRAM_VEC_SIZE, PROGRAM_KEY_VEC_SIZE, FIELD_WIDTH
from npi.add.lib import AdditionProgramSet, AdditionEnv, run_npi, create_questions, AdditionTeacher, \
    create_random_questions
from npi.core import NPIStep, Program, IntegerArguments, StepOutput, RuntimeSystem, PG_RETURN, StepInOut, StepInput, \
    to_one_hot_array
from npi.terminal_core import TerminalNPIRunner

__author__ = 'k_morishita'


class AdditionNPIModel(NPIStep):
    model = None
    f_enc = None

    def __init__(self, system: RuntimeSystem, model_path: str=None, program_set: AdditionProgramSet=None):
        self.system = system
        self.model_path = model_path
        self.program_set = program_set
        self.batch_size = 1
        self.build()
        self.weight_loaded = False
        self.load_weights()

    def build(self):
        enc_size = self.size_of_env_observation()
        argument_size = IntegerArguments.size_of_arguments
        input_enc = InputLayer(batch_input_shape=(self.batch_size, enc_size), name='input_enc')
        input_arg = InputLayer(batch_input_shape=(self.batch_size, argument_size), name='input_arg')
        input_prg = Embedding(input_dim=PROGRAM_VEC_SIZE, output_dim=PROGRAM_KEY_VEC_SIZE, input_length=1,
                              batch_input_shape=(self.batch_size, 1))

        f_enc = Sequential(name='f_enc')
        f_enc.add(Merge([input_enc, input_arg], mode='concat'))
        f_enc.add(MaxoutDense(128, nb_feature=4))
        self.f_enc = f_enc

        program_embedding = Sequential(name='program_embedding')
        program_embedding.add(input_prg)

        f_enc_convert = Sequential(name='f_enc_convert')
        f_enc_convert.add(f_enc)
        f_enc_convert.add(RepeatVector(1))

        f_lstm = Sequential(name='f_lstm')
        f_lstm.add(Merge([f_enc_convert, program_embedding], mode='concat'))
        f_lstm.add(LSTM(256, return_sequences=False, stateful=True, W_regularizer=l2(0.0000001)))
        f_lstm.add(Activation('relu', name='relu_lstm_1'))
        f_lstm.add(RepeatVector(1))
        f_lstm.add(LSTM(256, return_sequences=False, stateful=True, W_regularizer=l2(0.0000001)))
        f_lstm.add(Activation('relu', name='relu_lstm_2'))
        # plot(f_lstm, to_file='f_lstm.png', show_shapes=True)

        f_end = Sequential(name='f_end')
        f_end.add(f_lstm)
        f_end.add(Dense(1, W_regularizer=l2(0.001)))
        f_end.add(Activation('sigmoid', name='sigmoid_end'))

        f_prog = Sequential(name='f_prog')
        f_prog.add(f_lstm)
        f_prog.add(Dense(PROGRAM_KEY_VEC_SIZE, activation="relu"))
        f_prog.add(Dense(PROGRAM_VEC_SIZE, W_regularizer=l2(0.0001)))
        f_prog.add(Activation('softmax', name='softmax_prog'))
        # plot(f_prog, to_file='f_prog.png', show_shapes=True)

        f_args = []
        for ai in range(1, IntegerArguments.max_arg_num+1):
            f_arg = Sequential(name='f_arg%s' % ai)
            f_arg.add(f_lstm)
            f_arg.add(Dense(IntegerArguments.depth, W_regularizer=l2(0.0001)))
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

    def compile_model(self, lr=0.0001, arg_weight=1.):
        arg_num = IntegerArguments.max_arg_num
        optimizer = Adam(lr=lr)
        loss = ['binary_crossentropy', 'categorical_crossentropy'] + ['categorical_crossentropy'] * arg_num
        self.model.compile(optimizer=optimizer, loss=loss, loss_weights=[0.25, 0.25] + [arg_weight] * arg_num)

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

        # self.print_weights()
        if not self.weight_loaded:
            self.train_f_enc(filter_question(lambda a, b: 10 <= a < 100 and 10 <= b < 100), epoch=100)
        self.f_enc.trainable = False

        self.update_learning_rate(0.0001)

        # q_type = "training questions of a+b < 10"
        # print(q_type)
        # pr = 0.8
        # all_ok = self.fit_to_subset(filter_question(lambda a, b: a+b < 10), pass_rate=pr)
        # print("%s is pass_rate >= %s: %s" % (q_type, pr, all_ok))
        #
        # q_type = "training questions of a<10 and b< 10 and 10 <= a+b"
        # print(q_type)
        # pr = 0.8
        # all_ok = self.fit_to_subset(filter_question(lambda a, b: a<10 and b<10 and a + b >= 10), pass_rate=pr)
        # print("%s is pass_rate >= %s: %s" % (q_type, pr, all_ok))
        #
        # q_type = "training questions of a<10 and b<10"
        # print(q_type)
        # pr = 0.8
        # all_ok = self.fit_to_subset(filter_question(lambda a, b: a < 10 and b < 10), pass_rate=pr)
        # print("%s is pass_rate >= %s: %s" % (q_type, pr, all_ok))

        q_type = "training questions of a<100 and b<100"
        print(q_type)
        pr = 0.8
        all_ok = self.fit_to_subset(filter_question(lambda a, b: a < 100 and b < 100), pass_rate=pr)
        print("%s is pass_rate >= %s: %s" % (q_type, pr, all_ok))

        while True:
            if self.test_and_learn([10, 100, 1000]):
                break

            q_type = "training questions of ALL"
            print(q_type)

            q_num = 100
            skip_correct = False
            pr = 1.0
            questions = filter_question(lambda a, b: True)
            np.random.shuffle(questions)
            questions = questions[:q_num]
            all_ok = self.fit_to_subset(questions, pass_rate=pr, skip_correct=skip_correct)
            print("%s is pass_rate >= %s: %s" % (q_type, pr, all_ok))

    def fit_to_subset(self, steps_list, pass_rate=1.0, skip_correct=False):
        for i in range(10):
            all_ok = self.do_learn(steps_list, 100, pass_rate=pass_rate, skip_correct=skip_correct)
            if all_ok:
                return True
        return False

    def test_and_learn(self, num_questions):
        for num in num_questions:
            print("test all type of %d questions" % num)
            cc, wc, wrong_questions = self.test_to_subset(create_random_questions(num))
            acc_rate = cc/(cc+wc)
            print("Accuracy %s(OK=%d, NG=%d)" % (acc_rate, cc, wc))
            if wc > 0:
                self.fit_to_subset(wrong_questions, pass_rate=1.0, skip_correct=False)
                return False
        return True

    def test_to_subset(self, questions):
        addition_env = AdditionEnv(FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH)
        teacher = AdditionTeacher(self.program_set)
        npi_runner = TerminalNPIRunner(None, self)
        teacher_runner = TerminalNPIRunner(None, teacher)
        correct_count = wrong_count = 0
        wrong_steps_list = []
        for idx, question in enumerate(questions):
            question = copy(question)
            if self.question_test(addition_env, npi_runner, question):
                correct_count += 1
            else:
                self.question_test(addition_env, teacher_runner, question)
                wrong_steps_list.append({"q": question, "steps": teacher_runner.step_list})
                wrong_count += 1
        return correct_count, wrong_count, wrong_steps_list

    @staticmethod
    def dict_to_str(d):
        return str(tuple([(k, d[k]) for k in sorted(d)]))

    def do_learn(self, steps_list, epoch, pass_rate=1.0, skip_correct=False):
        addition_env = AdditionEnv(FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH)
        npi_runner = TerminalNPIRunner(None, self)
        last_weights = None
        correct_count = Counter()
        no_change_count = 0
        last_loss = 1000
        for ep in range(1, epoch+1):
            correct_new = wrong_new = 0
            losses = []
            ok_rate = []
            np.random.shuffle(steps_list)
            for idx, steps_dict in enumerate(steps_list):
                question = copy(steps_dict['q'])
                question_key = self.dict_to_str(question)
                if self.question_test(addition_env, npi_runner, question):
                    if correct_count[question_key] == 0:
                        correct_new += 1
                    correct_count[question_key] += 1
                    print("GOOD!: ep=%2d idx=%3d :%s CorrectCount=%s" % (ep, idx, self.dict_to_str(question), correct_count[question_key]))
                    ok_rate.append(1)
                    cc = correct_count[question_key]
                    if skip_correct or int(math.sqrt(cc)) ** 2 != cc:
                        continue
                else:
                    ok_rate.append(0)
                    if correct_count[question_key] > 0:
                        print("Degraded: ep=%2d idx=%3d :%s CorrectCount=%s -> 0" % (ep, idx, self.dict_to_str(question), correct_count[question_key]))
                        correct_count[question_key] = 0
                        wrong_new += 1

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
                    if not np.isfinite(loss):
                        print("Loss is not finite!, Last Input=%s" % ([i, (x, y, w)]))
                        self.print_weights(last_weights, detail=True)
                        raise RuntimeError("Loss is not finite!")
                    losses.append(loss)
                    last_weights = self.model.get_weights()
            if losses:
                cur_loss = np.average(losses)
                print("ep=%2d: ok_rate=%.2f%% (+%s -%s): ave loss %s (%s samples)" %
                      (ep, np.average(ok_rate)*100, correct_new, wrong_new, cur_loss, len(steps_list)))
                # self.print_weights()
                if correct_new + wrong_new == 0:
                    no_change_count += 1
                else:
                    no_change_count = 0

                if math.fabs(1 - cur_loss/last_loss) < 0.001 and no_change_count > 5:
                    print("math.fabs(1 - cur_loss/last_loss) < 0.001 and no_change_count > 5:")
                    return False
                last_loss = cur_loss
                print("=" * 80)
            self.save()
            if np.average(ok_rate) >= pass_rate:
                return True
        return False

    def update_learning_rate(self, learning_rate, arg_weight=1.):
        print("Re-Compile Model lr=%s aw=%s" % (learning_rate, arg_weight))
        self.compile_model(learning_rate, arg_weight=arg_weight)

    def train_f_enc(self, steps_list, epoch=50):
        print("training f_enc")
        f_add0 = Sequential(name='f_add0')
        f_add0.add(self.f_enc)
        f_add0.add(Dense(FIELD_DEPTH))
        f_add0.add(Activation('softmax', name='softmax_add0'))

        f_add1 = Sequential(name='f_add1')
        f_add1.add(self.f_enc)
        f_add1.add(Dense(FIELD_DEPTH))
        f_add1.add(Activation('softmax', name='softmax_add1'))

        env_model = Model(self.f_enc.inputs, [f_add0.output, f_add1.output], name="env_model")
        env_model.compile(optimizer='adam', loss=['categorical_crossentropy']*2)

        for ep in range(epoch):
            losses = []
            for idx, steps_dict in enumerate(steps_list):
                prev = None
                for step in steps_dict['steps']:
                    x = self.convert_input(step.input)[:2]
                    env_values = step.input.env.reshape((4, -1))
                    in1 = np.clip(env_values[0].argmax() - 1, 0, 9)
                    in2 = np.clip(env_values[1].argmax() - 1, 0, 9)
                    carry = np.clip(env_values[2].argmax() - 1, 0, 9)
                    y_num = in1 + in2 + carry
                    now = (in1, in2, carry)
                    if prev == now:
                        continue
                    prev = now
                    y0 = to_one_hot_array((y_num %  10)+1, FIELD_DEPTH)
                    y1 = to_one_hot_array((y_num // 10)+1, FIELD_DEPTH)
                    y = [yy.reshape((self.batch_size, -1)) for yy in [y0, y1]]
                    loss = env_model.train_on_batch(x, y)
                    losses.append(loss)
            print("ep %3d: loss=%s" % (ep, np.average(losses)))
            if np.average(losses) < 1e-06:
                break

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
            self.weight_loaded = True

    def print_weights(self, weights=None, detail=False):
        weights = weights or self.model.get_weights()
        for w in weights:
            print("w%s: sum(w)=%s, ave(w)=%s" % (w.shape, np.sum(w), np.average(w)))
        if detail:
            for w in weights:
                print("%s: %s" % (w.shape, w))

    @staticmethod
    def size_of_env_observation():
        return FIELD_ROW * FIELD_DEPTH
