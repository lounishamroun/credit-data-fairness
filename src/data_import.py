from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144)  
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
X.to_csv('features.csv')
y.to_csv('labels.csv')