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

class Scenario(Enum):
    OPEN_OPPORTUNITY = "open"  # Folded to
    FACING_UTG_RAISE = "vs_utg"
    FACING_MP_RAISE = "vs_mp"
    FACING_CO_RAISE = "vs_co"
    FACING_BTN_RAISE = "vs_btn"
    FACING_SB_RAISE = "vs_sb"

class Action(Enum):
    FOLD = "fold"
    CALL = "call"  # Changed from LIMP since we're facing raises
    RAISE = "raise"  # This becomes a 3-bet when facing a raise

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

def create_default_action_dict():
    """Create default action frequencies dictionary."""
    return {'fold': 1.0, 'call': 0.0, 'raise': 0.0}

def create_gto_charts():
    """
    Create GTO charts for each position and scenario.
    Returns nested dict: position -> scenario -> hand -> action frequencies
    """
    # UTG base strategy when open raising
    utg_chart_no_raise = {
        'AA': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'AKs': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'AQs': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'AJs': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'ATs': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'A9s': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'A8s': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'A7s': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'A6s': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'A5s': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'A4s': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'A3s': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'A2s': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'AKo': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'KK': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'KQs': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'KJs': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'KTs': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'K9s': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'K8s': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'K7s': {'fold': 0.47040000000000004, 'call': 0.0, 'raise': 0.5296},
        'K6s': {'fold': 0.3496, 'call': 0.0, 'raise': 0.6504},
        'K5s': {'fold': 0.00029999999999996696, 'call': 0.0, 'raise': 0.9997},
        'K4s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'K3s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'K2s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'AQo': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'KQo': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'QQ': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'QJs': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'QTs': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'Q9s': {'fold': 0.9958, 'call': 0.0, 'raise': 0.0042},
        'Q8s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q7s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q6s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q5s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q4s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q3s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q2s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'AJo': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'KJo': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'QJo': {'fold': 0.7101, 'call': 0.0, 'raise': 0.2899},
        'JJ': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'JTs': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'J9s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J8s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J7s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J6s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J5s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J4s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J3s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J2s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'ATo': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'KTo': {'fold': 0.6864, 'call': 0.0, 'raise': 0.3136},
        'QTo': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'JTo': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'TT': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'T9s': {'fold': 0.06589999999999996, 'call': 0.0, 'raise': 0.9341},
        'T8s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T7s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T6s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T5s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T4s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T3s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T2s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'A9o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'K9o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q9o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J9o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T9o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '99': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        '98s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '97s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '96s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '95s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '94s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '93s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '92s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'A8o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'K8o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q8o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J8o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T8o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '98o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '88': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        '87s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '86s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '85s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '84s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '83s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '82s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'A7o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'K7o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q7o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J7o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T7o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '97o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '87o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '77': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        '76s': {'fold': 0.7741, 'call': 0.0, 'raise': 0.2259},
        '75s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '74s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '73s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '72s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'A6o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'K6o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q6o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J6o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T6o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '96o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '86o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '76o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '66': {'fold': 0.5449999999999999, 'call': 0.0, 'raise': 0.455},
        '65s': {'fold': 0.31920000000000004, 'call': 0.0, 'raise': 0.6808},
        '64s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '63s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '62s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'A5o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'K5o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q5o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J5o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T5o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '95o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '85o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '75o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '65o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '55': {'fold': 0.5001, 'call': 0.0, 'raise': 0.4999},
        '54s': {'fold': 0.917, 'call': 0.0, 'raise': 0.083},
        '53s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '52s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'A4o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'K4o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q4o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J4o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T4o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '94o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '84o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '74o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '64o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '54o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '44': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '43s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '42s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'A3o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'K3o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q3o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J3o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T3o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '93o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '83o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '73o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '63o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '53o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '43o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '33': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '32s': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'A2o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'K2o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'Q2o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'J2o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        'T2o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '92o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '82o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '72o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '62o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '52o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '42o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '32o': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
        '22': {'fold': 1.0, 'call': 0.0, 'raise': 0.0}
 }

    # MP vs UTG raise
    mp_vs_utg = {
        'AA': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'KK': {'fold': 0.0, 'call': 0.0, 'raise': 1.0},
        'QQ': {'fold': 0.0, 'call': 0.2, 'raise': 0.8},
        'JJ': {'fold': 0.0, 'call': 0.8, 'raise': 0.2},
        # Add more hands...
    }

    # Create the nested dictionary structure
    charts = {}
    
    # Add default action distributions for all hands in each scenario
    all_hands = create_hand_matrix()
    default_actions = create_default_action_dict()
    
    # UTG position charts
    charts[Position.UTG] = {
        Scenario.OPEN_OPPORTUNITY: {hand: default_actions.copy() for hand in all_hands}
    }
    charts[Position.UTG][Scenario.OPEN_OPPORTUNITY].update(utg_chart_no_raise)
    
    # MP position charts
    charts[Position.MP] = {
        Scenario.OPEN_OPPORTUNITY: {hand: default_actions.copy() for hand in all_hands},
        Scenario.FACING_UTG_RAISE: {hand: default_actions.copy() for hand in all_hands}
    }
    charts[Position.MP][Scenario.FACING_UTG_RAISE].update(mp_vs_utg)
    
    # Add similar structures for other positions...
    
    return charts

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

def determine_scenario(row: pd.Series, position: Position) -> Scenario:
    """Determine the scenario based on previous actions."""
    positions_order = ['utg', 'mp', 'co', 'btn', 'sb', 'bb']
    pos_index = positions_order.index(position.value)
    previous_positions = positions_order[:pos_index]
    
    # Check previous actions
    for prev_pos in previous_positions:
        action = row[f'preflop_action_{prev_pos}']
        if pd.notna(action) and 'raise' in action:
            return getattr(Scenario, f'FACING_{prev_pos.upper()}_RAISE')
    
    return Scenario.OPEN_OPPORTUNITY

def analyze_player_actions(df: pd.DataFrame, player_name: str, gto_charts: Dict) -> Dict:
    """Analyze how closely a player's actions match GTO charts."""
    results = {}
    
    for position in Position:
        # Filter hands where player was in this position
        position_hands = df[df[position.value] == player_name]
        
        if len(position_hands) == 0:
            continue
            
        # Group hands by scenario
        scenario_results = {}
        for _, hand in position_hands.iterrows():
            scenario = determine_scenario(hand, position)
            
            if scenario not in scenario_results:
                scenario_results[scenario] = {}
                
            standardized_hand = hand['standardized_hand']
            if standardized_hand not in scenario_results[scenario]:
                scenario_results[scenario][standardized_hand] = {
                    'total_hands': 0,
                    'fold_frequency': 0,
                    'call_frequency': 0,
                    'raise_frequency': 0
                }
                
            stats = scenario_results[scenario][standardized_hand]
            stats['total_hands'] += 1
            
            # Determine action taken
            action = hand[f'preflop_action_{position.value}']
            if pd.isna(action):
                continue
                
            if 'fold' in action:
                stats['fold_frequency'] += 1
            elif 'call' in action:
                stats['call_frequency'] += 1
            elif 'raise' in action:
                stats['raise_frequency'] += 1
                
        # Convert counts to frequencies and calculate deviations
        for scenario in scenario_results:
            for hand in scenario_results[scenario]:
                stats = scenario_results[scenario][hand]
                total = stats['total_hands']
                if total > 0:
                    stats['fold_frequency'] /= total
                    stats['call_frequency'] /= total
                    stats['raise_frequency'] /= total
                    
                # Compare to GTO frequencies
                if position in gto_charts and scenario in gto_charts[position]:
                    gto_freqs = gto_charts[position][scenario].get(
                        hand, {'fold': 1.0, 'call': 0.0, 'raise': 0.0}
                    )
                    stats['gto_fold_freq'] = gto_freqs['fold']
                    stats['gto_call_freq'] = gto_freqs['call']
                    stats['gto_raise_freq'] = gto_freqs['raise']
                    
                    # Calculate deviations
                    stats['fold_deviation'] = abs(stats['fold_frequency'] - gto_freqs['fold'])
                    stats['call_deviation'] = abs(stats['call_frequency'] - gto_freqs['call'])
                    stats['raise_deviation'] = abs(stats['raise_frequency'] - gto_freqs['raise'])
                    stats['total_deviation'] = (
                        stats['fold_deviation'] + 
                        stats['call_deviation'] + 
                        stats['raise_deviation']
                    ) / 3
                
        results[position.value] = scenario_results
    
    return results

def generate_report(analysis_results: Dict) -> str:
    """Generate a readable report from the analysis results."""
    report = "GTO Analysis Report\n==================\n\n"
    
    for position, scenarios in analysis_results.items():
        report += f"\nPosition: {position.upper()}\n{'-' * (len(position) + 10)}\n"
        
        if not scenarios:
            report += "No hands played in this position\n"
            continue
            
        for scenario, hands in scenarios.items():
            report += f"\nScenario: {scenario.value}\n"
            
            # Calculate overall statistics for this scenario
            total_hands = sum(stats['total_hands'] for stats in hands.values())
            if total_hands > 0:
                weighted_deviation = sum(
                    stats.get('total_deviation', 0) * stats['total_hands'] 
                    for stats in hands.values()
                ) / total_hands
            else:
                weighted_deviation = 0
            
            report += f"Total hands: {total_hands}\n"
            report += f"Overall GTO deviation: {weighted_deviation:.1%}\n\n"
            
            # Sort hands by deviation
            sorted_hands = sorted(
                hands.items(),
                key=lambda x: x[1].get('total_deviation', 0),
                reverse=True
            )
            
            # Show most significant deviations
            report += "Most significant deviations:\n"
            for hand, stats in sorted_hands[:5]:
                if stats.get('total_deviation', 0) > 0.05:
                    report += f"\nHand: {hand}\n"
                    report += f"  Played {stats['total_hands']} times\n"
                    report += f"  Actual frequencies: Fold={stats['fold_frequency']:.1%}, "
                    report += f"Call={stats['call_frequency']:.1%}, "
                    report += f"Raise={stats['raise_frequency']:.1%}\n"
                    if 'gto_fold_freq' in stats:
                        report += f"  GTO frequencies: Fold={stats['gto_fold_freq']:.1%}, "
                        report += f"Call={stats['gto_call_freq']:.1%}, "
                        report += f"Raise={stats['gto_raise_freq']:.1%}\n"
    
    return report

def main():
    """Main function to run the analysis."""
    try:
        # Configuration
        file_path = "hand_history.txt"  # Update this to your file path
        player_name = "Vadims"          # Update this to the player you want to analyze
        
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