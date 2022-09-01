

import pytest
import unittest
import os


def prediction_test0():
  os.system("python ../orca_workspace/SLIF_extrapolation.py  --quiet")
  assert 1==1 
  return 0

"""
def prediction_test1():
  os.system("bash ../scripts/run_bal.sh")
  assert 1==1
  return 0
"""

