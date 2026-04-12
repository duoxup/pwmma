#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 20:04:19 2026

@author: duoxup
"""

from numpy import pi, cos, sin

def phase_parity(x: int):
    x_int = int(x)
    sgn = -1.0 if (x_int % 2) else 1.0
    return sgn

def intcc1(a1, a2, m1, m2, x1, x2):
    if m2 == m1 and m1 == 0:
        return a1
    elif m2*a1 == m1*a2 and m2*m1 != 0:
        return a1/2 * cos(m2 * pi / a2 * x1)
    else:
        return m2 * a2 * a1**2 / (pi * ((m2 * a1)**2 - (m1 * a2)**2)) *\
            (phase_parity(m1) * sin(m2 * pi / a2 * x2) - sin(m2 * pi / a2 * x1))

def intcc2(b1, b2, n1, n2, y1, y2):
    if n2 == n1 and n1 == 0:
        return b1
    elif n2*b1 == n1*b2 and n2*n1 != 0:
        return b1/2 * cos(n2 * pi / b2 * y1)
    else:
        return n2 * b2 * b1**2 / (pi * ((n2 * b1)**2 - (n1 * b2)**2)) *\
            (phase_parity(n1) * sin(n2 * pi / b2 * y2) - sin(n2 * pi / b2 * y1))

def intss1(a1, a2, m1, m2, x1, x2):
    if m2*a1 == m1*a2:
        return a1/2 * cos(m2 * pi / a2 * x1)
    else:
        return m1 * a2**2 * a1 / (pi * ((m2 * a1)**2 - (m1 * a2)**2)) *\
            (phase_parity(m1) * sin(m2 * pi / a2 * x2) - sin(m2 * pi / a2 * x1))

def intss2(b1, b2, n1, n2, y1, y2):
    if n2*b1 == n1*b2:
        return b1/2 * cos(n2 * pi / b2 * y1)
    else:
        return n1 * b2**2 * b1 / (pi * ((n2 * b1)**2 - (n1 * b2)**2)) *\
            (phase_parity(n1) * sin(n2 * pi / b2 * y2) - sin(n2 * pi / b2 * y1))

def xys_concentric(a1, b1, a2, b2, x1, x2, y1, y2):
    x1 = a2/2 - a1/2 if x1 is None else x1
    x2 = a2/2 + a1/2 if x2 is None else x2
    y1 = b2/2 - b1/2 if y1 is None else y1
    y2 = b2/2 + b1/2 if y2 is None else y2
    return x1, x2, y1, y2
    

def hh(a1, b1, a2, b2, m1, n1, m2, n2, kc_mn1, kc_mn2, norm_factor,
       x1=None, x2=None, y1=None, y2=None):
    x1, x2, y1, y2 = xys_concentric(a1, b1, a2, b2, x1, x2, y1, y2)
    return norm_factor * kc_mn1**2 * intcc1(a1, a2, m1, m2, x1, x2) *\
        intcc2(b1, b2, n1, n2, y1, y2)

def eh(*args, **kwargs):
    return 0.

def he(a1, b1, a2, b2, m1, n1, m2, n2, kc_mn1, kc_mn2, norm_factor,
       x1=None, x2=None, y1=None, y2=None):
    x1, x2, y1, y2 = xys_concentric(a1, b1, a2, b2, x1, x2, y1, y2)
    return norm_factor * (m1 * pi / a1 * (phase_parity(n1) * sin(n2 * pi / b2 * y2) - sin(n2 * pi / b2 * y1)) * intss1(a1, a2, m1, m2, x1, x2) \
                          -n1 * pi / b1 * (phase_parity(m1) * sin(m2 * pi / a2 * x2) - sin(m2 * pi / a2 * x1)) * intss2(b1, b2, n1, n2, y1, y2))

def ee(a1, b1, a2, b2, m1, n1, m2, n2, kc_mn1, kc_mn2, norm_factor,
       x1=None, x2=None, y1=None, y2=None):
    x1, x2, y1, y2 = xys_concentric(a1, b1, a2, b2, x1, x2, y1, y2)
    return norm_factor * kc_mn2**2 * intss1(a1, a2, m1, m2, x1, x2) *\
        intss2(b1, b2, n1, n2, y1, y2)