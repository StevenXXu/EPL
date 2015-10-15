
def get_non_feature_columns():
    """ Returns a list of the columns that are in our features dataframe that
        should not be used in prediction. These are essentially either metadata
        columns (team name, for example), or potential target variables that
        include the outcome. We want to make sure not to use the latter, since
        we don't want to use information about the current game to predict that
        same game.
    """
    return ['teamid', 'op_teamid', 'matchid', 'competitionid', 'seasonid',
            'goals', 'op_goals', 'points', 'timestamp', 'team_name', 
            'op_team_name']

def get_feature_columns(all_cols):
    """ Returns a list of all columns that should be used in prediction
        (i.e. all features that are in the dataframe but are not in the 
        features.get_non_feature_column() list).
    """
    return [col for col in all_cols if col not in get_non_feature_columns()]

