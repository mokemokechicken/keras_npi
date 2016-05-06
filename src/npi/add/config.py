#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'
FIELD_ROW = 4     # Input1, Input2, Carry, Output
FIELD_WIDTH = 9   # number of columns
FIELD_DEPTH = 11  # number of characters(0~9 digits) and white space, per cell. one-hot-encoding
PROGRAM_VEC_SIZE = 10
KEY_VEC_SIZE = 5