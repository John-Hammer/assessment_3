import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple

class Position(Enum):
    UTG = "utg"
    MP = "mp"
    CO = "co"
    BTN = "btn"
    SB = "sb"
    BB = "bb"

class Action(Enum):
    FOLD = "fold"
    LIMP = "call"
    RAISE = "raise"

# Define card ranks and suits
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'o']  # s for suited, o for offsuit

def create_hand_matrix():
    """Create a matrix of all possible starting hands."""
    hands = []
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i > j:  # Suited combinations
                hands.append(f"{r1}{r2}s")
            elif i < j:  # Offsuit combinations
                hands.append(f"{r1}{r2}o")
            elif i == j:  # Pocket pairs
                hands.append(f"{r1}{r1}")
    return sorted(hands, key=lambda x: (RANKS.index(x[0]), RANKS.index(x[1])))

def create_gto_charts():
    """
    Create GTO charts for each position.
    Values represent the percentage of time to raise (0.0 to 1.0)
    For example: 0.85 means raise 85% of the time, fold 15%
    """
    # UTG chart based on the image (purple = raise, coral = fold, mixed shown by ratio)
    utg_chart_no_raise = {
        'AA': 1.0, 'AKs': 1.0, 'AQs': 1.0, 'AJs': 1.0, 'ATs': 1.0, 'A9s': 1.0, 'A8s': 1.0, 'A7s': 1.0, 'A6s': 1.0, 'A5s': 1.0, 'A4s': 1.0, 'A3s': 1.0, 'A2s': 1.0,
        'AKo': 1.0, 'KK': 1.0, 'KQs': 1.0, 'KJs': 1.0, 'KTs': 1.0, 'K9s': 1.0, 'K8s': 1.0, 'K7s': 0.5296, 'K6s': 0.6504, 'K5s': 0.9997, 'K4s': 0.0, 'K3s': 0.0, 'K2s': 0.0,
        'AQo': 1.0, 'KQo': 1.0, 'QQ': 1.0, 'QJs': 1.0, 'QTs': 1.0, 'Q9s': 0.0042, 'Q8s': 0.0, 'Q7s': 0.0, 'Q6s': 0.0, 'Q5s': 0.0, 'Q4s': 0.0, 'Q3s': 0.0, 'Q2s': 0.0,
        'AJo': 1.0, 'KJo': 1.0, 'QJo': 0.2899, 'JJ': 1.0, 'JTs': 1.0, 'J9s': 0.0, 'J8s': 0.0, 'J7s': 0.0, 'J6s': 0.0, 'J5s': 0.0, 'J4s': 0.0, 'J3s': 0.0, 'J2s': 0.0,
        'ATo': 1.0, 'KTo': 0.3136, 'QTo': 0.0, 'JTo': 0.0, 'TT': 1.0, 'T9s': 0.9341, 'T8s': 0.0, 'T7s': 0.0, 'T6s': 0.0, 'T5s': 0.0, 'T4s': 0.0, 'T3s': 0.0, 'T2s': 0.0,
        'A9o': 0.0, 'K9o': 0.0, 'Q9o': 0.0, 'J9o': 0.0, 'T9o': 0.0, '99': 1.0, '98s': 0.0, '97s': 0.0, '96s': 0.0, '95s': 0.0, '94s': 0.0, '93s': 0.0, '92s': 0.0,
        'A8o': 0.0, 'K8o': 0.0, 'Q8o': 0.0, 'J8o': 0.0, 'T8o': 0.0, '98o': 0.0, '88': 1.0, '87s': 0.0, '86s': 0.0, '85s': 0.0, '84s': 0.0, '83s': 0.0, '82s': 0.0,
        'A7o': 0.0, 'K7o': 0.0, 'Q7o': 0.0, 'J7o': 0.0, 'T7o': 0.0, '97o': 0.0, '87o': 0.0, '77': 1.0, '76s': 0.2259, '75s': 0.0, '74s': 0.0, '73s': 0.0, '72s': 0.0,
        'A6o': 0.0, 'K6o': 0.0, 'Q6o': 0.0, 'J6o': 0.0, 'T6o': 0.0, '96o': 0.0, '86o': 0.0, '76o': 0.0, '66': 0.455, '65s': 0.6808, '64s': 0.0, '63s': 0.0, '62s': 0.0,
        'A5o': 0.0, 'K5o': 0.0, 'Q5o': 0.0, 'J5o': 0.0, 'T5o': 0.0, '95o': 0.0, '85o': 0.0, '75o': 0.0, '65o': 0.0, '55': 0.4999, '54s': 0.083, '53s': 0.0, '52s': 0.0,
        'A4o': 0.0, 'K4o': 0.0, 'Q4o': 0.0, 'J4o': 0.0, 'T4o': 0.0, '94o': 0.0, '84o': 0.0, '74o': 0.0, '64o': 0.0, '54o': 0.0, '44': 0.0, '43s': 0.0, '42s': 0.0,
        'A3o': 0.0, 'K3o': 0.0, 'Q3o': 0.0, 'J3o': 0.0, 'T3o': 0.0, '93o': 0.0, '83o': 0.0, '73o': 0.0, '63o': 0.0, '53o': 0.0, '43o': 0.0, '33': 0.0, '32s': 0.0,
        'A2o': 0.0, 'K2o': 0.0, 'Q2o': 0.0, 'J2o': 0.0, 'T2o': 0.0, '92o': 0.0, '82o': 0.0, '72o': 0.0, '62o': 0.0, '52o': 0.0, '42o': 0.0, '32o': 0.0, '22': 0.0
    }

    hj_chart_no_raise = {
        'AA': 1.0, 'AKs': 1.0, 'AQs': 1.0, 'AJs': 1.0, 'ATs': 1.0, 'A9s': 1.0, 'A8s': 1.0, 'A7s': 1.0, 'A6s': 1.0, 'A5s': 1.0, 'A4s': 1.0, 'A3s': 1.0, 'A2s': 1.0,
        'AKo': 1.0, 'KK': 1.0, 'KQs': 1.0, 'KJs': 1.0, 'KTs': 1.0, 'K9s': 1.0, 'K8s': 1.0, 'K7s': 1.0, 'K6s': 1.0, 'K5s': 1.0, 'K4s': 0.0001, 'K3s': 0.0, 'K2s': 0.0,
        'AQo': 1.0, 'KQo': 1.0, 'QQ': 1.0, 'QJs': 1.0, 'QTs': 1.0, 'Q9s': 1.0, 'Q8s': 0.0001, 'Q7s': 0.0, 'Q6s': 0.0, 'Q5s': 0.0, 'Q4s': 0.0, 'Q3s': 0.0, 'Q2s': 0.0,
        'AJo': 1.0, 'KJo': 1.0, 'QJo': 1.0, 'JJ': 1.0, 'JTs': 1.0, 'J9s': 0.9992, 'J8s': 0.0, 'J7s': 0.0, 'J6s': 0.0, 'J5s': 0.0, 'J4s': 0.0, 'J3s': 0.0, 'J2s': 0.0,
        'ATo': 1.0, 'KTo': 1.0, 'QTo': 0.447, 'JTo': 0.0, 'TT': 1.0, 'T9s': 1.0, 'T8s': 0.7146, 'T7s': 0.0, 'T6s': 0.0, 'T5s': 0.0, 'T4s': 0.0, 'T3s': 0.0, 'T2s': 0.0,
        'A9o': 0.7221, 'K9o': 0.0, 'Q9o': 0.0, 'J9o': 0.0, 'T9o': 0.0, '99': 1.0, '98s': 0.0, '97s': 0.0, '96s': 0.0, '95s': 0.0, '94s': 0.0, '93s': 0.0, '92s': 0.0,
        'A8o': 0.0021, 'K8o': 0.0, 'Q8o': 0.0, 'J8o': 0.0, 'T8o': 0.0, '98o': 0.0, '88': 1.0, '87s': 0.0, '86s': 0.0, '85s': 0.0, '84s': 0.0, '83s': 0.0, '82s': 0.0,
        'A7o': 0.0, 'K7o': 0.0, 'Q7o': 0.0, 'J7o': 0.0, 'T7o': 0.0, '97o': 0.0, '87o': 0.0, '77': 1.0, '76s': 0.1505, '75s': 0.0, '74s': 0.0, '73s': 0.0, '72s': 0.0,
        'A6o': 0.0, 'K6o': 0.0, 'Q6o': 0.0, 'J6o': 0.0, 'T6o': 0.0, '96o': 0.0, '86o': 0.0, '76o': 0.0, '66': 1.0, '65s': 0.9294, '64s': 0.0, '63s': 0.0, '62s': 0.0,
        'A5o': 0.151, 'K5o': 0.0, 'Q5o': 0.0, 'J5o': 0.0, 'T5o': 0.0, '95o': 0.0, '85o': 0.0, '75o': 0.0, '65o': 0.0, '55': 1.0, '54s': 0.139, '53s': 0.0, '52s': 0.0,
        'A4o': 0.0, 'K4o': 0.0, 'Q4o': 0.0, 'J4o': 0.0, 'T4o': 0.0, '94o': 0.0, '84o': 0.0, '74o': 0.0, '64o': 0.0, '54o': 0.0, '44': 0.0, '43s': 0.0, '42s': 0.0,
        'A3o': 0.0, 'K3o': 0.0, 'Q3o': 0.0, 'J3o': 0.0, 'T3o': 0.0, '93o': 0.0, '83o': 0.0, '73o': 0.0, '63o': 0.0, '53o': 0.0, '43o': 0.0, '33': 0.0, '32s': 0.0,
        'A2o': 0.0, 'K2o': 0.0, 'Q2o': 0.0, 'J2o': 0.0, 'T2o': 0.0, '92o': 0.0, '82o': 0.0, '72o': 0.0, '62o': 0.0, '52o': 0.0, '42o': 0.0, '32o': 0.0, '22': 0.0
    }

    co_chart_no_raise = {
        'AA': 1.0, 'AKs': 1.0, 'AQs': 1.0, 'AJs': 1.0, 'ATs': 1.0, 'A9s': 1.0, 'A8s': 1.0, 'A7s': 1.0, 'A6s': 1.0, 'A5s': 1.0, 'A4s': 1.0, 'A3s': 1.0, 'A2s': 1.0,
        'AKo': 1.0, 'KK': 1.0, 'KQs': 1.0, 'KJs': 1.0, 'KTs': 1.0, 'K9s': 1.0, 'K8s': 1.0, 'K7s': 1.0, 'K6s': 1.0, 'K5s': 1.0, 'K4s': 1.0, 'K3s': 0.9999, 'K2s': 0.0,
        'AQo': 1.0, 'KQo': 1.0, 'QQ': 1.0, 'QJs': 1.0, 'QTs': 1.0, 'Q9s': 1.0, 'Q8s': 1.0, 'Q7s': 0.9972, 'Q6s': 0.9958, 'Q5s': 0.074, 'Q4s': 0.0, 'Q3s': 0.0, 'Q2s': 0.0,
        'AJo': 1.0, 'KJo': 1.0, 'QJo': 1.0, 'JJ': 1.0, 'JTs': 1.0, 'J9s': 1.0, 'J8s': 1.0, 'J7s': 0.0, 'J6s': 0.0, 'J5s': 0.0, 'J4s': 0.0, 'J3s': 0.0, 'J2s': 0.0,
        'ATo': 1.0, 'KTo': 1.0, 'QTo': 1.0, 'JTo': 1.0, 'TT': 1.0, 'T9s': 1.0, 'T8s': 1.0, 'T7s': 0.9985, 'T6s': 0.0, 'T5s': 0.0, 'T4s': 0.0, 'T3s': 0.0, 'T2s': 0.0,
        'A9o': 1.0, 'K9o': 0.0001, 'Q9o': 0.0, 'J9o': 0.0, 'T9o': 0.0, '99': 1.0, '98s': 1.0, '97s': 0.6057, '96s': 0.0, '95s': 0.0, '94s': 0.0, '93s': 0.0, '92s': 0.0,
        'A8o': 0.9999, 'K8o': 0.0, 'Q8o': 0.0, 'J8o': 0.0, 'T8o': 0.0, '98o': 0.0, '88': 1.0, '87s': 0.3947, '86s': 0.0, '85s': 0.0, '84s': 0.0, '83s': 0.0, '82s': 0.0,
        'A7o': 0.0, 'K7o': 0.0, 'Q7o': 0.0, 'J7o': 0.0, 'T7o': 0.0, '97o': 0.0, '87o': 0.0, '77': 1.0, '76s': 0.6347, '75s': 0.0, '74s': 0.0, '73s': 0.0, '72s': 0.0,
        'A6o': 0.0, 'K6o': 0.0, 'Q6o': 0.0, 'J6o': 0.0, 'T6o': 0.0, '96o': 0.0, '86o': 0.0, '76o': 0.0, '66': 1.0, '65s': 0.9993, '64s': 0.0, '63s': 0.0, '62s': 0.0,
        'A5o': 0.9892, 'K5o': 0.0, 'Q5o': 0.0, 'J5o': 0.0, 'T5o': 0.0, '95o': 0.0, '85o': 0.0, '75o': 0.0, '65o': 0.0, '55': 1.0, '54s': 0.4368, '53s': 0.0, '52s': 0.0,
        'A4o': 0.0, 'K4o': 0.0, 'Q4o': 0.0, 'J4o': 0.0, 'T4o': 0.0, '94o': 0.0, '84o': 0.0, '74o': 0.0, '64o': 0.0, '54o': 0.0, '44': 1.0, '43s': 0.0, '42s': 0.0,
        'A3o': 0.0, 'K3o': 0.0, 'Q3o': 0.0, 'J3o': 0.0, 'T3o': 0.0, '93o': 0.0, '83o': 0.0, '73o': 0.0, '63o': 0.0, '53o': 0.0, '43o': 0.0, '33': 0.0465, '32s': 0.0,
        'A2o': 0.0, 'K2o': 0.0, 'Q2o': 0.0, 'J2o': 0.0, 'T2o': 0.0, '92o': 0.0, '82o': 0.0, '72o': 0.0, '62o': 0.0, '52o': 0.0, '42o': 0.0, '32o': 0.0, '22': 0.0
    }

    btn_chart_no_raise = {
        'AA': 1.0, 'AKs': 1.0, 'AQs': 1.0, 'AJs': 1.0, 'ATs': 1.0, 'A9s': 1.0, 'A8s': 1.0, 'A7s': 1.0, 'A6s': 1.0, 'A5s': 1.0, 'A4s': 1.0, 'A3s': 1.0, 'A2s': 1.0,
        'AKo': 1.0, 'KK': 1.0, 'KQs': 1.0, 'KJs': 1.0, 'KTs': 1.0, 'K9s': 1.0, 'K8s': 1.0, 'K7s': 1.0, 'K6s': 1.0, 'K5s': 1.0, 'K4s': 1.0, 'K3s': 1.0, 'K2s': 1.0,
        'AQo': 1.0, 'KQo': 1.0, 'QQ': 1.0, 'QJs': 1.0, 'QTs': 1.0, 'Q9s': 1.0, 'Q8s': 1.0, 'Q7s': 1.0, 'Q6s': 1.0, 'Q5s': 1.0, 'Q4s': 1.0, 'Q3s': 1.0, 'Q2s': 1.0,
        'AJo': 1.0, 'KJo': 1.0, 'QJo': 1.0, 'JJ': 1.0, 'JTs': 1.0, 'J9s': 1.0, 'J8s': 1.0, 'J7s': 1.0, 'J6s': 1.0, 'J5s': 1.0, 'J4s': 0.9977, 'J3s': 0.0, 'J2s': 0.0,
        'ATo': 1.0, 'KTo': 1.0, 'QTo': 1.0, 'JTo': 1.0, 'TT': 1.0, 'T9s': 1.0, 'T8s': 1.0, 'T7s': 1.0, 'T6s': 1.0, 'T5s': 0.2758, 'T4s': 0.0, 'T3s': 0.0, 'T2s': 0.0,
        'A9o': 1.0, 'K9o': 1.0, 'Q9o': 1.0, 'J9o': 1.0, 'T9o': 1.0, '99': 1.0, '98s': 1.0, '97s': 1.0, '96s': 1.0, '95s': 0.0, '94s': 0.0, '93s': 0.0, '92s': 0.0,
        'A8o': 1.0, 'K8o': 0.9768, 'Q8o': 0.3131, 'J8o': 0.0, 'T8o': 0.6307, '98o': 0.0, '88': 1.0, '87s': 1.0, '86s': 0.9998, '85s': 0.0, '84s': 0.0, '83s': 0.0, '82s': 0.0,
        'A7o': 1.0, 'K7o': 0.3599, 'Q7o': 0.0, 'J7o': 0.0, 'T7o': 0.0, '97o': 0.0, '87o': 0.0, '77': 1.0, '76s': 1.0, '75s': 1.0, '74s': 0.0, '73s': 0.0, '72s': 0.0,
        'A6o': 1.0, 'K6o': 0.0, 'Q6o': 0.0, 'J6o': 0.0, 'T6o': 0.0, '96o': 0.0, '86o': 0.0, '76o': 0.0, '66': 1.0, '65s': 1.0, '64s': 0.0, '63s': 0.0, '62s': 0.0,
        'A5o': 1.0, 'K5o': 0.0, 'Q5o': 0.0, 'J5o': 0.0, 'T5o': 0.0, '95o': 0.0, '85o': 0.0, '75o': 0.0, '65o': 0.0, '55': 1.0, '54s': 0.9994, '53s': 0.0, '52s': 0.0,
        'A4o': 1.0, 'K4o': 0.0, 'Q4o': 0.0, 'J4o': 0.0, 'T4o': 0.0, '94o': 0.0, '84o': 0.0, '74o': 0.0, '64o': 0.0, '54o': 0.0, '44': 1.0, '43s': 0.0, '42s': 0.0,
        'A3o': 1.0, 'K3o': 0.0, 'Q3o': 0.0, 'J3o': 0.0, 'T3o': 0.0, '93o': 0.0, '83o': 0.0, '73o': 0.0, '63o': 0.0, '53o': 0.0, '43o': 0.0, '33': 1.0, '32s': 0.0,
        'A2o': 0.0, 'K2o': 0.0, 'Q2o': 0.0, 'J2o': 0.0, 'T2o': 0.0, '92o': 0.0, '82o': 0.0, '72o': 0.0, '62o': 0.0, '52o': 0.0, '42o': 0.0, '32o': 0.0, '22': 0.9997
    }

    sb_chart_no_raise = {
        'AA': 1.0, 'AKs': 1.0, 'AQs': 1.0, 'AJs': 1.0, 'ATs': 1.0, 'A9s': 1.0, 'A8s': 1.0, 'A7s': 1.0, 'A6s': 1.0, 'A5s': 1.0, 'A4s': 1.0, 'A3s': 1.0, 'A2s': 1.0,
        'AKo': 1.0, 'KK': 1.0, 'KQs': 1.0, 'KJs': 1.0, 'KTs': 1.0, 'K9s': 1.0, 'K8s': 1.0, 'K7s': 1.0, 'K6s': 1.0, 'K5s': 1.0, 'K4s': 1.0, 'K3s': 1.0, 'K2s': 1.0,
        'AQo': 1.0, 'KQo': 1.0, 'QQ': 1.0, 'QJs': 1.0, 'QTs': 1.0, 'Q9s': 1.0, 'Q8s': 1.0, 'Q7s': 1.0, 'Q6s': 1.0, 'Q5s': 1.0, 'Q4s': 1.0, 'Q3s': 1.0, 'Q2s': 1.0,
        'AJo': 1.0, 'KJo': 1.0, 'QJo': 1.0, 'JJ': 1.0, 'JTs': 1.0, 'J9s': 1.0, 'J8s': 1.0, 'J7s': 1.0, 'J6s': 1.0, 'J5s': 1.0, 'J4s': 0.3614, 'J3s': 0.0, 'J2s': 0.0,
        'ATo': 1.0, 'KTo': 1.0, 'QTo': 1.0, 'JTo': 1.0, 'TT': 1.0, 'T9s': 1.0, 'T8s': 1.0, 'T7s': 1.0, 'T6s': 1.0, 'T5s': 0.0007, 'T4s': 0.0, 'T3s': 0.0, 'T2s': 0.0,
        'A9o': 1.0, 'K9o': 1.0, 'Q9o': 1.0, 'J9o': 1.0, 'T9o': 1.0, '99': 1.0, '98s': 1.0, '97s': 1.0, '96s': 1.0, '95s': 0.0002, '94s': 0.0, '93s': 0.0, '92s': 0.0,
        'A8o': 1.0, 'K8o': 0.4472, 'Q8o': 0.0, 'J8o': 0.0, 'T8o': 0.3829, '98o': 0.6425, '88': 1.0, '87s': 1.0, '86s': 1.0, '85s': 1.0, '84s': 0.0, '83s': 0.0, '82s': 0.0,
        'A7o': 1.0, 'K7o': 0.0, 'Q7o': 0.0, 'J7o': 0.0, 'T7o': 0.0, '97o': 0.0, '87o': 0.0, '77': 1.0, '76s': 1.0, '75s': 1.0, '74s': 0.0002, '73s': 0.0, '72s': 0.0,
        'A6o': 1.0, 'K6o': 0.0, 'Q6o': 0.0, 'J6o': 0.0, 'T6o': 0.0, '96o': 0.0, '86o': 0.0, '76o': 0.0, '66': 1.0, '65s': 1.0, '64s': 1.0, '63s': 0.0, '62s': 0.0,
        'A5o': 1.0, 'K5o': 0.0, 'Q5o': 0.0, 'J5o': 0.0, 'T5o': 0.0, '95o': 0.0, '85o': 0.0, '75o': 0.0, '65o': 0.0, '55': 1.0, '54s': 1.0, '53s': 0.7582, '52s': 0.0,
        'A4o': 1.0, 'K4o': 0.0, 'Q4o': 0.0, 'J4o': 0.0, 'T4o': 0.0, '94o': 0.0, '84o': 0.0, '74o': 0.0, '64o': 0.0, '54o': 0.0, '44': 1.0, '43s': 0.0, '42s': 0.0,
        'A3o': 0.9999, 'K3o': 0.0, 'Q3o': 0.0, 'J3o': 0.0, 'T3o': 0.0, '93o': 0.0, '83o': 0.0, '73o': 0.0, '63o': 0.0, '53o': 0.0, '43o': 0.0, '33': 1.0, '32s': 0.0,
        'A2o': 0.0, 'K2o': 0.0, 'Q2o': 0.0, 'J2o': 0.0, 'T2o': 0.0, '92o': 0.0, '82o': 0.0, '72o': 0.0, '62o': 0.0, '52o': 0.0, '42o': 0.0, '32o': 0.0, '22': 1.0
    }
        
    # For all other hands not explicitly listed, default to 0.0 (fold)
    all_hands = create_hand_matrix()
    for hand in all_hands:
        if hand not in utg_chart:
            utg_chart[hand] = 0.0
    
    # For now, we'll use more aggressive charts for later positions
    # In practice, you'd want to create specific charts for each position
    mp_chart = {hand: min(1.0, value * 1.2) for hand, value in utg_chart.items()}
    co_chart = {hand: min(1.0, value * 1.4) for hand, value in utg_chart.items()}
    btn_chart = {hand: min(1.0, value * 1.6) for hand, value in utg_chart.items()}
    sb_chart = {hand: min(1.0, value * 1.3) for hand, value in utg_chart.items()}
    bb_chart = utg_chart  # BB plays differently due to being last to act
    
    return {
        Position.UTG: utg_chart,
        Position.MP: mp_chart,
        Position.CO: co_chart,
        Position.BTN: btn_chart,
        Position.SB: sb_chart,
        Position.BB: bb_chart,
    }

def parse_hand_history(file_path: str) -> pd.DataFrame:
    """Parse the hand history file into a DataFrame."""
    # Read the hand history file
    df = pd.read_csv(file_path, delimiter=' ')
    
    # Convert hole cards to standardized format
    def standardize_hand(cards: str) -> str:
        if not isinstance(cards, str):
            return ''
        card1, card2 = cards.split()
        rank1, rank2 = card1[0], card2[0]
        suited = card1[1] == card2[1]
        if rank1 == rank2:
            return f"{rank1}{rank1}"
        elif suited:
            return f"{max(rank1, rank2)}{min(rank1, rank2)}s"
        else:
            return f"{max(rank1, rank2)}{min(rank1, rank2)}o"
    
    df['standardized_hand'] = df['hole_cards'].apply(standardize_hand)
    return df

def calculate_gto_error(actual_freq: float, gto_freq: float) -> str:
    """Calculate and categorize GTO deviation."""
    deviation = actual_freq - gto_freq
    if abs(deviation) < 0.05:
        return "Optimal (±5%)"
    elif abs(deviation) < 0.10:
        return f"{'Over' if deviation > 0 else 'Under'}-aggressive (±10%)"
    else:
        return f"Significant {'over' if deviation > 0 else 'under'}-play ({deviation:+.1%})"

def analyze_player_actions(df: pd.DataFrame, player_name: str, gto_charts: Dict) -> Dict:
    """Analyze how closely a player's actions match GTO charts."""
    results = {}
    
    for position in Position:
        # Filter hands where player was in this position
        position_hands = df[df[position.value] == player_name]
        
        if len(position_hands) == 0:
            continue
            
        hand_stats = {}
        for hand in create_hand_matrix():
            hand_instances = position_hands[position_hands['standardized_hand'] == hand]
            
            if len(hand_instances) == 0:
                continue
                
            # Calculate action frequencies
            total_hands = len(hand_instances)
            fold_freq = len(hand_instances[hand_instances[f'preflop_action_{position.value}'].str.contains('fold', na=False)]) / total_hands
            raise_freq = len(hand_instances[hand_instances[f'preflop_action_{position.value}'].str.contains('raise', na=False)]) / total_hands
            limp_freq = len(hand_instances[hand_instances[f'preflop_action_{position.value}'].str.contains('call', na=False)]) / total_hands
            
            # Compare to GTO chart
            gto_raise_freq = gto_charts[position].get(hand, 0.0)
            gto_deviation = abs(raise_freq - gto_raise_freq)
            
            hand_stats[hand] = {
                'total_hands': total_hands,
                'fold_frequency': fold_freq,
                'raise_frequency': raise_freq,
                'limp_frequency': limp_freq,
                'gto_raise_frequency': gto_raise_freq,
                'gto_deviation': gto_deviation,
                'gto_error': calculate_gto_error(raise_freq, gto_raise_freq)
            }
            
        results[position.value] = hand_stats
    
    return results

def generate_report(analysis_results: Dict) -> str:
    """Generate a readable report from the analysis results."""
    report = "GTO Analysis Report\n==================\n\n"
    
    for position, hands in analysis_results.items():
        report += f"\nPosition: {position.upper()}\n{'-' * (len(position) + 10)}\n"
        
        if not hands:
            report += "No hands played in this position\n"
            continue
        
        # Sort hands by GTO deviation for easier reading
        sorted_hands = sorted(hands.items(), key=lambda x: x[1]['gto_deviation'], reverse=True)
        
        # Overall statistics
        total_hands = sum(stats['total_hands'] for _, stats in sorted_hands)
        weighted_deviation = sum(stats['gto_deviation'] * stats['total_hands'] 
                               for _, stats in sorted_hands) / total_hands if total_hands > 0 else 0
        
        report += f"Total hands: {total_hands}\n"
        report += f"Overall GTO deviation: {weighted_deviation:.1%}\n\n"
        
        # Individual hand analysis
        report += "Most significant deviations:\n"
        for hand, stats in sorted_hands[:10]:  # Show top 10 deviations
            if stats['gto_deviation'] > 0.05:  # Only show meaningful deviations
                report += f"\nHand: {hand}\n"
                report += f"  Played {stats['total_hands']} times\n"
                report += f"  Actual raise frequency: {stats['raise_frequency']:.1%}\n"
                report += f"  GTO raise frequency: {stats['gto_raise_frequency']:.1%}\n"
                report += f"  Assessment: {stats['gto_error']}\n"
    
    return report

def main():
    """Main function to run the analysis."""
    try:
        # Configuration
        file_path = "./archive/wpn_blitz_10nl.csv"  # Update this to your file path
        player_name = "Vadims" # Update this to the player you want to analyze
        
        # Create GTO charts
        print("Creating GTO charts...")
        gto_charts = create_gto_charts()
        
        # Parse hand history
        print(f"Parsing hand history from {file_path}...")
        df = parse_hand_history(file_path)
        
        # Analyze player actions
        print(f"Analyzing actions for player: {player_name}...")
        analysis_results = analyze_player_actions(df, player_name, gto_charts)
        
        # Generate and print report
        print("\nGenerating report...\n")
        report = generate_report(analysis_results)
        print(report)
        
        # Optionally save report to file
        with open("gto_analysis_report.txt", "w") as f:
            f.write(report)
        print("\nReport saved to gto_analysis_report.txt")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()