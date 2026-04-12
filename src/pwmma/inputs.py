#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 21:45:40 2026

@author: duoxup
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import Sequence

from waveguides import WG

@dataclass
class Transition:
    wg1: WG
    wg2: WG
    
    def __post_init__(self) -> None:
        if not isinstance(self.wg1, WG):
            raise TypeError(f'\'wg1\' has to be a WG instance, got {type(self.wg1)}')
        if not isinstance(self.wg2, WG):
            raise TypeError(f'\'wg2\' has to be a WG instance, got {type(self.wg2)}')
            
    def swap(self):
        return Transition(self.wg2, self.wg1)
            
@dataclass
class Chain:
    wgs: Sequence[WG]
    sym: bool = False
    
    def __post_init__(self) -> None:
        for idx, wg in enumerate(self.wgs):
            if not isinstance(wg, WG):
                raise TypeError(f'All elements have to be WG instances, wgs[{idx}] got {type(self.wgs[idx])}')
        if self.n_wgs < 2:
            raise ValueError('A Chain instance has to contain at least 2 waveguides')
          
    @property        
    def n_wgs(self) -> int:
        return len(self.wgs)
    
    @property 
    def transitions(self) -> Sequence[Transition]:
        transs = []
        for idx in range(self.n_wgs-1):
            transs.append(Transition(self.wgs[idx], self.wgs[idx+1]))
        return transs
                
                
    
    
    
    
    