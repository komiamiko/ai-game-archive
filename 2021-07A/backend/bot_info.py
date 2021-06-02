"""
Contains basic info on all of the bots.
Please register your bots by adding them in this file.
"""

import typing

class BotInfo(object):
    """
    Profile information for a bot.
    For contestants adding bots, please do not override the rating.
    
    Each command is passed to the shell, using the subprocess library.
    The last command is intended to be the actual bot.
    That process will remain alive for the duration of a match.
    Previous commands may be used for preparation or initialization.
    
    Since you may specify any command for the bot, it is possible to instantiate
    multiple versions of the same bot with different parameters for tuning or testing.
    Please do not submit excessive variations of the same bot for the official competition.
    
    If you are submitting multiple versions of the same bot or very closely related bots,
    please also specify the family argument to indicate the bots are from the same family.
    This is relevant for tournament ranking purposes.
    """
    def __init__(self, name: str, author: str,
        commands: list[list[str]], family: typing.Optional[str] = None,
        rating: typing.Optional['Player'] = None):
        self.name = name
        self.author = author
        self.commands = commands
        self.family = family
        self.rating = rating
    def __str__(self):
        return self.name
    def __eq__(self, other: 'BotInfo'):
        return self.name == other.name
    def __ne__(self, other: 'BotInfo'):
        return not (self == other)
    def __hash__(self):
        return hash((BotInfo, self.name))

_all_bots = None

def get_all_bots() -> list[BotInfo]:
    """
    Get all bots in the tournament.
    Contestants, please add your bots here.
    """
    global _all_bots
    if _all_bots is not None:
        return _all_bots
    all_bots = []
    
    _all_bots = all_bots
    return all_bots
