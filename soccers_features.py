#!/usr/bin/python2.7
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    Turns raw statistics about soccer matches into features we use
    for prediction. Combines a number of games of history to compute
    aggregates that can be used to predict the next game.
"""

import numpy as np
import pandas as pd

# Games that have stats available. Not all games in the match_games table
# will have stats (e.g. they might be in the future).

# Combines statistics from both teams in a match.
# For each two records matching the pattern (m, t1, <stats1>) and
# (m, t2, <stats2>) where m is the match id, t1 and t2 are the two teams,
# stats1 and stats2 are the statistics for those two teams, combines them
# into a single row (m, t1, t2, <stats1>, <stats2>) where all of the
# t2 field names are decorated with the op_ prefix. For example, teamid becomes
# op_teamid, and pass_70 becomes op_pass_70.


def get_match_history(history_size): 
    """ For each team t in each game g, computes the N previous game 
        ids where team t played, where N is the history_size (number
        of games of history we use for prediction). The statistics of
        the N previous games will be used to predict the outcome of 
        game g.
    """
    return 

def get_history_query(history_size): 
    """ Computes summary statistics for the N preceeding matches. """
    return 

def get_history_query_with_goals(history_size):
    """ Expands the history_query, which summarizes statistics from past games
        with the result of who won the current game. This information will not
        be availble for future games that we want to predict, but it will be
        available for past games. We can then use this information to train our
        models.
    """
    return 

def get_wc_history_query(history_size): 
    """ Identical to the history_query (which, remember, does not have
        outcomes) but gets history for world-cup games.
    """
    return 

def get_wc_features(history_size):
    """ Runs a bigquery query that gets the features that can be used
        to predict the world cup.
    """
    return 

def get_features(history_size):
    """ Runs a BigQuery query to get features that can be used to train
         a machine learning model.
    """
    return 

def get_game_summaries():
    """ Runs a BigQuery Query that gets game summaries. """
    return 
