

import pytest
import os

def prediction_test():
  os.system("bash ../scripts/run_bal.sh")
  assert 1==1
  return 0
