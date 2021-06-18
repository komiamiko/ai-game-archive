"""
Main tournament runner for ShipTurret.
"""

import argparse
import pathlib
import subprocess
from ..backend.bot_info import BotInfo, get_all_bots
from .glicko2 import Player
from .tournament_db import Database

VERBOSE_MINIMUM = 0
VERBOSE_MORE = 1
VERBOSE_ALL = 2
VERBOSE_DEBUG = 3
VERBOSE_MAP = {
    'minimum': VERBOSE_MINIMUM,
    'more': VERBOSE_MORE,
    'all': VERBOSE_ALL,
    'debug': VERBOSE_DEBUG,
    }

def get_options():
    """
    Get options from the command line.
    """
    parser = argparse.ArgumentParser(description='Run one phase of the tournament for ShipTurret.')
    parser.add_argument('-b', '--block', type=int, default=8,
                        help='Maximum number of matches to run before updating scores and synchronizing with ' + \
                             'the disk database.')
    parser.add_argument('-t', '--threads', type=int, default=4,
                        help='Maximum number of matches to have running in parallel. Note that one match may ' + \
                             'produce more compute stress than 1 concurrent thread\'s worth, especially if ' + \
                             'bots continue running in the background when not on their turn or use extra threads.')
    parser.add_argument('-p', '--phase', choices=('opening', 'main', 'finals', 'report'),
                        required=True,
                        help='Which phase of the tournament to run. Opening, main, and finals phase progress' + \
                             'the tournament. Report generates the report files.')
    parser.add_argument('-v', '--verbose', choices=('minimum', 'more', 'all', 'debug'),
                        default='more',
                        help='How much information to print during program execution. Minimum prints only key ' + \
                             'information. More prints a moderate amount of information. All prints a lot of ' + \
                             'detail, and often. Debug will further include debugging information.')
    parser.add_argument('-d', '--rating-bracket', type=int, default=1000,
                     help='Maximum rating difference for matchmaking during the main phase. If running ' + \
                          'multiple main phases, it is recommended to decrease the bracket each time.')
    options = parser.parse_args()
    if options.block <= 0:
        print('Block size should be positive.')
        exit(1)
    if options.threads <= 0:
        print('Max threads should be positive.')
        exit(1)
    options.verbose_level = VERBOSE_MAP[options.verbose]
    return options

def main():
    """
    Main function.
    """
    options = get_options()


if __name__ == '__main__':
    main()
