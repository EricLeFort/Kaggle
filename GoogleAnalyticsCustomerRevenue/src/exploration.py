#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import pandas as pd
import json

def extractDeviceType(device):
	return json.loads(device)['operatingSystem']

print('Loading data.. ', end='', flush=True)
train = pd.read_csv(
	'../data/train.csv',
	date_parser = lambda x: pd.datetime.strptime(x, '%Y%m%d'),
	parse_dates=['date'],
	dtype={
		'channelGrouping': 'category',
		#'date': 'datetime64', inferred using parse_dates
		'device': object,
		'fullVisitorId': object,
		'geoNetwork': object,
		'sessionId': object,
		'socialEngagementType': 'category',
		'totals': object,
		'trafficSource': object,
		'visitId': 'int64',
		'visitNumber': 'int64',
		'visitStartTime': 'int64'
	}
)
#This field is invariant (only "Not Socially Engaged"), not needed
train.drop(columns=['channelGrouping'])
print('Done.')

print('Extracting operatingSystem from JSON.. ', end='', flush=True)
#Extract transactionRevenue from totals, drop that JSON
train['operatingSystem'] = train['device'].map(extractDeviceType)
train = train.drop(columns=['device'])
print('Done')

print(train['operatingSystem'].unique())

# Total Count: 903653
#print('Total Count: ' + str(train.shape[0]))

# Unique Ids: 714167
#print('Unique Ids: ' + str(train['fullVisitorId'].unique().shape[0]))

# First-Visit Transactions: 7050
# Non-First-Visit Transactions: 4465
#transactionsMade = train[train['totals'].str.contains('transactionRevenue')]
#print('First-Visit Transactions: ' + str(transactionsMade[transactionsMade['visitNumber'] > 1].shape[0]))
#print('Non-First-Visit Transactions: ' + str(transactionsMade[transactionsMade['visitNumber'] == 1].shape[0]))

# Values of socialEngagementType: [Not Socially Engaged]
#print(train['socialEngagementType'].unique())

# Values of channelGrouping: [Organic Search, Referral, Paid Search, Affiliates, Direct, Display, Social, (Other)]
#print(train['channelGrouping'].unique())

# Distribution of channelGrouping:
#(Other)             120 
#Affiliates        16403
#Direct           143026
#Display            6262
#Organic Search   381561
#Paid Search       25326
#Referral         104838
#Social           226117

# 21,600 visitors visited using different channels
#channels = train.groupby(['fullVisitorId', 'channelGrouping']).size() \
#	.reset_index() \
#	.rename(columns={0:'count'})
#print(channels.shape)
#channels = channels.drop_duplicates(subset='fullVisitorId')
#print(channels.shape)


# Tricks Explored:
# Simply rounding low scores to 0 did not work at all (attempted with Gradient boosting)

