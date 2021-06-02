"""
Null bot - does nothing for the entire game.
"""

while True:
    s = input()
    if s == 'end_game':
        break
    if s == 'end_tick':
        print('end_action')