# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 21:37:00 2015

@author: Steven
"""

import power
import soccer_prediction
import match_stats
reload(power)
reload(soccer_prediction)

import pandas as pd

club_data = pd.read_csv('EPL_dataset.csv')

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Don't train on games that ended in a draw, since they have less signal.
train = club_data.loc[club_data['points'] <> 1] 
# train = club_data

(model, test) = soccer_prediction.train_model(
     train, match_stats.get_non_feature_columns())
print "\nRsquared: %0.03g" % model.prsquared

def points_to_sgn(p):
  if p > 0.1: return 1.0
  elif p < -0.1: return -1.0
  else: return 0.0
power_cols = [
  ('points', points_to_sgn, 'points'),
]

power_data = power.add_power(club_data, club_data, power_cols)
power_train = power_data.loc[power_data['points'] <> 1] 

# power_train = power_data
(power_model, power_test) = soccer_prediction.train_model(
    power_train, match_stats.get_non_feature_columns())
print "\nRsquared: %0.03g, Power Coef %0.03g" % (
    power_model.prsquared, 
    math.exp(power_model.params['power_points']))

power_results = soccer_prediction.predict_model(power_model, power_test, 
    match_stats.get_non_feature_columns())
power_y = [yval == 3 for yval in power_test['points']]
soccer_prediction.validate(3, power_y, power_results['predicted'], baseline, 
                   compute_auc=True, quiet=False)

pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
# Add the old model to the graph
world_cup.validate('old', y, results['predicted'], baseline, 
                   compute_auc=True, quiet=True)
pl.legend(loc="lower right")
pl.show()

print_params(power_model, 8)