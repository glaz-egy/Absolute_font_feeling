# -*- coding: utf-8 -*-

import math
import sys
import time

END = 170
MAX_LEN = 30

def get_progressber_str(progress):
    BAR_LEN = int(MAX_LEN * progress)
    return('[' + '=' * BAR_LEN + ('>' if BAR_LEN < MAX_LEN else '') +
            ' ' * (MAX_LEN - BAR_LEN) + '] %.1f%%' % (progress * 100.))
