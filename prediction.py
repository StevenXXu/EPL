import soccer_prediction
#reload(soccer_prediction)
import match_stats
import pandas as pd
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

club_data = pd.read_csv('EPL_dataset.csv')

# Don't train on games that ended in a draw, since they have less signal.
train = club_data.loc[club_data['points'] <> 1] 
# train = club_data

(model, test) = soccer_prediction.train_model(
     train, match_stats.get_non_feature_columns())
print "\nRsquared: %0.03g" % model.prsquared