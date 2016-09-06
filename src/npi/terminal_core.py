#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
import curses
import numpy as np

from npi.core import Program, IntegerArguments, NPIStep, StepOutput, StepInput, StepInOut

__author__ = u'k_morishita'


class Screen(object):
    data = None

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.init_screen()

    def init_screen(self):
        self.data = np.zeros([self.height, self.width], dtype=np.int8)

    def fill(self, ch):
        self.data.fill(ch)

    def as_float32(self):
        return self.data.astype(np.float32)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, item):
        return self.data[item]


class Terminal(object):
    W_TOP = 1
    W_LEFT = 1
    LOG_WINDOW_HEIGHT = 10
    LOG_WINDOW_WIDTH = 80
    INFO_WINDOW_HEIGHT = 10
    INFO_WINDOW_WIDTH = 40

    main_window = None
    info_window = None
    log_window = None

    def __init__(self, stdscr, char_map=None):
        print type(stdscr)
        self.stdscr = stdscr
        self.char_map = char_map or dict((ch, unichr(ch)) for ch in xrange(128))
        self.log_list = []

    def init_window(self, width, height):
        curses.curs_set(0)
        border_win = curses.newwin(height + 2, width + 2, self.W_TOP, self.W_LEFT)  # h, w, y, x
        border_win.box()
        self.stdscr.refresh()
        border_win.refresh()
        self.main_window = curses.newwin(height, width, self.W_TOP + 1, self.W_LEFT + 1)
        self.main_window.refresh()
        self.main_window.timeout(1)
        self.info_window = curses.newwin(self.INFO_WINDOW_HEIGHT, self.INFO_WINDOW_WIDTH,
                                         self.W_TOP + 1, self.W_LEFT + width + 2)
        self.log_window = curses.newwin(self.LOG_WINDOW_HEIGHT, self.LOG_WINDOW_WIDTH,
                                        self.W_TOP + max(height, self.INFO_WINDOW_HEIGHT) + 5, self.W_LEFT)
        self.log_window.refresh()

    def wait_for_key(self):
        self.stdscr.getch()

    def update_main_screen(self, screen):
        for y in xrange(screen.height):
            line = u"".join([self.char_map[ch] for ch in screen[y]])
            self.ignore_error_add_str(self.main_window, y, 0, line)

    def update_main_window_attr(self, screen, y, x, attr):
        ch = screen[y, x]
        self.ignore_error_add_str(self.main_window, y, x, self.char_map[ch], attr)

    def refresh_main_window(self):
        self.main_window.refresh()

    def update_info_screen(self, info_list):
        self.info_window.clear()
        for i, info_str in enumerate(info_list):
            self.info_window.addstr(i, 2, info_str)
        self.info_window.refresh()

    def add_log(self, line):
        self.log_list.insert(0, unicode(line)[:self.LOG_WINDOW_WIDTH])
        self.log_list = self.log_list[:self.LOG_WINDOW_HEIGHT-1]
        self.log_window.clear()
        for i, line in enumerate(self.log_list):
            line = unicode(line) + u" " * (self.LOG_WINDOW_WIDTH - len(unicode(line)))
            self.log_window.addstr(i, 0, line)
        self.log_window.refresh()

    @staticmethod
    def ignore_error_add_str(win, y, x, s, attr=curses.A_NORMAL):
        u"""一番右下に書き込むと例外が飛んでくるけど、漢は黙って無視するのがお作法らしい？"""
        try:
            win.addstr(y, x, s, attr)
        except curses.error:
            pass


def show_env_to_terminal(terminal, env):
    terminal.update_main_screen(env.screen)
    for i, p in enumerate(env.pointers):
        terminal.update_main_window_attr(env.screen, i, p, curses.A_REVERSE)
    terminal.refresh_main_window()


class TerminalNPIRunner(object):
    def __init__(self, terminal, model=None, recording=True, max_depth=10, max_step=1000):
        self.terminal = terminal
        self.model = model
        self.steps = 0
        self.step_list = []
        self.alpha = 0.5
        self.verbose = True
        self.recording = recording
        self.max_depth = max_depth
        self.max_step = max_step

    def reset(self):
        self.steps = 0
        self.step_list = []
        self.model.reset()

    def display_env(self, env, force=False):
        if (self.verbose or force) and self.terminal:
            show_env_to_terminal(self.terminal, env)

    def display_information(self, program, arguments, result, depth):
        if self.verbose and self.terminal:
            information = [
                u"Step %2d Depth: %2d" % (self.steps, depth),
                program.description_with_args(arguments),
                u'r=%.2f' % result.r,
            ]
            if result.program:
                information.append(u"-> %s" % result.program.description_with_args(result.arguments))
            self.terminal.update_info_screen(information)
            self.wait()

    def npi_program_interface(self, env, program, arguments, depth=0):
        if self.max_depth < depth or self.max_step < self.steps:
            raise StopIteration()

        self.model.enter_function()

        result = StepOutput(0, None, None)
        while result.r < self.alpha:
            self.steps += 1
            if self.max_step < self.steps:
                raise StopIteration()

            env_observation = env.get_observation()
            result = self.model.step(env_observation, program, arguments.copy())
            if self.recording:
                self.step_list.append(StepInOut(StepInput(env_observation, program, arguments.copy()), result))
            self.display_information(program, arguments, result, depth)

            if program.output_to_env:
                program.do(env, arguments.copy())
                self.display_env(env)
            else:
                if result.program:  # modify original algorithm
                    self.npi_program_interface(env, result.program, result.arguments, depth=depth+1)

        self.model.exit_function()

    def wait(self):
        self.terminal.wait_for_key()
