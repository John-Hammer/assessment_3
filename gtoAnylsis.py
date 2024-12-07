import pandas as pd
import numpy as np
from collections import defaultdict

class GTODeviationAnalyzer:
    def __init__(self):
        # RFI (Raise First In) percentages from the charts
        self.rfi_frequencies = {
            'LJ': 17.0,    # 226/1326 hands
            'HJ': 21.4,    # 284/1326 hands
            'CO': 27.8,    # 368/1326 hands
            'BTN': 43.3,   # 574/1326 hands
            'SB': {
                'raise': 24.5,  # 322/1326 hands
                'limp': 36.0    # 504/1326 hands
            }
        }
        
        # Position mapping
        self.position_mapping = {
            'utg': 'LJ',
            'mp': 'HJ',
            'co': 'CO',
            'btn': 'BTN',
            'sb': 'SB',
            'bb': 'BB'
        }

    def _get_position(self, hand_data):
        """Determine the position for analysis based on preflop action"""
        # Check each position's action in order
        positions = ['utg', 'mp', 'co', 'btn', 'sb', 'bb']
        for pos in positions:
            action_col = f'preflop_action_{pos}'
            if pd.notna(hand_data[action_col]) and hand_data[action_col] != '':
                return self.position_mapping[pos]
        return None

    def _get_action(self, hand_data):
        """Extract the actual action taken from the hand data"""
        positions = ['utg', 'mp', 'co', 'btn', 'sb', 'bb']
        for pos in positions:
            action_col = f'preflop_action_{pos}'
            if pd.notna(hand_data[action_col]) and hand_data[action_col] != '':
                action = hand_data[action_col].lower()
                if 'raise' in action:
                    return 'raise'
                elif 'call' in action:
                    return 'call'
                elif 'fold' in action:
                    return 'fold'
                elif 'check' in action:
                    return 'check'
        return None

    def _get_facing_action(self, hand_data):
        """Determine what action the player was facing"""
        positions = ['utg', 'mp', 'co', 'btn', 'sb', 'bb']
        current_pos = None
        
        # Find the position being analyzed
        for pos in positions:
            action_col = f'preflop_action_{pos}'
            if pd.notna(hand_data[action_col]) and hand_data[action_col] != '':
                current_pos = positions.index(pos)
                break
                
        if current_pos is None or current_pos == 0:
            return 'none'
            
        # Check if any earlier position has raised
        for pos in positions[:current_pos]:
            action_col = f'preflop_action_{pos}'
            if pd.notna(hand_data[action_col]) and 'raise' in hand_data[action_col].lower():
                return 'raise'
            elif pd.notna(hand_data[action_col]) and 'call' in hand_data[action_col].lower():
                return 'call'
        
        return 'none'

    def _analyze_rfi_spot(self, position, action, hand_type):
        """Analyze a Raise First In situation"""
        gto_frequency = self.rfi_frequencies.get(position, 0)
        
        # Simplified analysis - could be expanded with actual hand ranges
        should_raise = (hand_type in self._get_rfi_range(position))
        
        return {
            'position': position,
            'situation': 'RFI',
            'hand': hand_type,
            'actual_play': action,
            'gto_play': 'raise' if should_raise else 'fold',
            'is_gto': (action == 'raise') == should_raise
        }

    def _analyze_vs_rfi_spot(self, position, facing_action, action, hand_type):
        """Analyze a facing RFI situation"""
        key = f"{position}_vs_{facing_action}"
        frequencies = self.vs_rfi_frequencies.get(key, {})
        
        # Simplified analysis - could be expanded with actual hand ranges
        should_3bet = (hand_type in self._get_3bet_range(position, facing_action))
        should_call = (hand_type in self._get_call_range(position, facing_action))
        
        gto_play = 'fold'
        if should_3bet:
            gto_play = '3bet'
        elif should_call:
            gto_play = 'call'
            
        return {
            'position': position,
            'situation': f'vs_{facing_action}',
            'hand': hand_type,
            'actual_play': action,
            'gto_play': gto_play,
            'is_gto': action == gto_play
        }

    def _get_rfi_range(self, position):
        """Get the RFI range for a position - placeholder for demonstration"""
        # This should be expanded with actual ranges from the charts
        return {'AKs', 'AQs', 'AKo', 'KQs'}  # Example range

    def _get_3bet_range(self, position, facing_action):
        """Get the 3bet range for a position - placeholder for demonstration"""
        # This should be expanded with actual ranges from the charts
        return {'AKs', 'AAs', 'KKs'}  # Example range

    def _get_call_range(self, position, facing_action):
        """Get the calling range for a position - placeholder for demonstration"""
        # This should be expanded with actual ranges from the charts
        return {'JTs', 'QJs', 'KQs'}  # Example range

    def analyze_hand(self, hand_data):
        """Analyze a single hand for GTO deviations"""
        position = self._get_position(hand_data)
        action = self._get_action(hand_data)
        facing_action = self._get_facing_action(hand_data)
        
        if not all([position, action]):
            return None
            
        hand_type = self._convert_hand_format(hand_data['hole_cards'])
        
        if facing_action == 'none':
            return self._analyze_rfi_spot(position, action, hand_type)
        else:
            return self._analyze_vs_rfi_spot(position, facing_action, action, hand_type)

    def _convert_hand_format(self, hole_cards):
        """Convert hole cards to standard notation (e.g., AKs, AKo)"""
        if pd.isna(hole_cards):
            return None
            
        cards = hole_cards.split()
        if len(cards) != 2:
            return None
            
        card1, card2 = cards[0], cards[1]
        rank1, suit1 = card1[0], card1[1]
        rank2, suit2 = card2[0], card2[1]
        
        suited = 's' if suit1 == suit2 else 'o'
        return f"{rank1}{rank2}{suited}"

    def process_hand_histories(self, df):
        """Process all hands and identify deviations"""
        deviations = []
        summaries = defaultdict(lambda: {'total': 0, 'deviations': 0, 'hands': []})
        
        for _, hand in df.iterrows():
            analysis = self.analyze_hand(hand)
            if analysis:  # Skip hands that couldn't be analyzed
                situation = f"{analysis['position']}_{analysis['situation']}"
                summaries[situation]['total'] += 1
                
                if not analysis['is_gto']:
                    deviations.append(analysis)
                    summaries[situation]['deviations'] += 1
                    summaries[situation]['hands'].append({
                        'hand': analysis['hand'],
                        'actual_play': analysis['actual_play'],
                        'gto_play': analysis['gto_play']
                    })

        return self._generate_report(deviations, summaries)

    def _generate_report(self, deviations, summaries):
        """Generate comprehensive analysis report"""
        report = {
            'overall_summary': {
                'total_hands': sum(s['total'] for s in summaries.values()),
                'total_deviations': len(deviations),
                'deviation_percentage': round(len(deviations) / max(sum(s['total'] for s in summaries.values()), 1) * 100, 2)
            },
            'position_summary': {},
            'biggest_leaks': [],
            'detailed_deviations': deviations
        }

        # Analyze position-specific patterns
        for situation, data in summaries.items():
            if data['total'] > 0:
                deviation_rate = (data['deviations'] / data['total']) * 100
                report['position_summary'][situation] = {
                    'total_hands': data['total'],
                    'deviations': data['deviations'],
                    'deviation_rate': round(deviation_rate, 2)
                }

        # Identify biggest leaks
        situation_deviations = defaultdict(list)
        for dev in deviations:
            key = f"{dev['position']}_{dev['situation']}"
            situation_deviations[key].append(dev)

        for situation, devs in situation_deviations.items():
            if len(devs) >= 3:  # Only include patterns with at least 3 occurrences
                report['biggest_leaks'].append({
                    'situation': situation,
                    'frequency': len(devs),
                    'example_hands': devs[:3]
                })

        return report

    def print_report(self, report):
        """Print formatted analysis report"""
        print("\n=== GTO Deviation Analysis Report ===\n")
        
        print("Overall Summary:")
        print(f"Total Hands Analyzed: {report['overall_summary']['total_hands']}")
        print(f"Total Deviations: {report['overall_summary']['total_deviations']}")
        print(f"Overall Deviation Rate: {report['overall_summary']['deviation_percentage']}%\n")
        
        print("Position-Specific Analysis:")
        for situation, data in report['position_summary'].items():
            print(f"\n{situation}:")
            print(f"  Hands: {data['total_hands']}")
            print(f"  Deviations: {data['deviations']}")
            print(f"  Deviation Rate: {data['deviation_rate']}%")
        
        print("\nBiggest Leaks:")
        for leak in report['biggest_leaks']:
            print(f"\n{leak['situation']}:")
            print(f"Frequency: {leak['frequency']} times")
            print("Example hands:")
            for hand in leak['example_hands']:
                print(f"  {hand['hand']}: Played {hand['actual_play']} instead of {hand['gto_play']}")

# Example usage
if __name__ == "__main__":
    file_path = "./archive/wpn_blitz_10nl.csv"
    try:
        hands_df = pd.read_csv(file_path)
        analyzer = GTODeviationAnalyzer()
        report = analyzer.process_hand_histories(hands_df)
        analyzer.print_report(report)
    except Exception as e:
        print(f"Error processing hand histories: {str(e)}")