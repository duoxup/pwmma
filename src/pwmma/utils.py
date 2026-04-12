#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:50:04 2026

@author: duoxup
"""
import math
from .inputs import Transition

def judge_cross_section_containment(wgt: Transition) -> int:
    """
    Return:
      1 -> wg1 is fully inside wg2
      2 -> wg2 is fully inside wg1
      0 -> same size (mutual containment)
    Raise:
      ValueError if neither fully contains the other.

    Assumptions:
      - wgt has .wg1 and .wg2
      - each wg has .cross_tag in {'rec','cir'}
      - rec: has .a, .b (full side lengths)
      - cir: has .r (radius)
      - same center, rectangles are axis-aligned
    """
    wg1, wg2 = wgt.wg1, wgt.wg2

    def contains(outer, inner) -> bool:
        if outer.cross_tag == "rec" and inner.cross_tag == "rec":
            return (inner.a <= outer.a) and (inner.b <= outer.b)

        if outer.cross_tag == "cir" and inner.cross_tag == "cir":
            return inner.r <= outer.r

        if outer.cross_tag == "rec" and inner.cross_tag == "cir":
            return inner.r <= 0.5 * min(outer.a, outer.b)

        if outer.cross_tag == "cir" and inner.cross_tag == "rec":
            return math.hypot(0.5 * inner.a, 0.5 * inner.b) <= outer.r

        raise ValueError(
            f"Unsupported cross_tag combination: "
            f"outer={outer.cross_tag!r}, inner={inner.cross_tag!r}"
        )

    w1_in_w2 = contains(wg2, wg1)
    w2_in_w1 = contains(wg1, wg2)

    if w1_in_w2 and w2_in_w1:
        return 0
    if w1_in_w2:
        return 1
    if w2_in_w1:
        return 2

    raise ValueError("Neither cross-section fully contains the other.")