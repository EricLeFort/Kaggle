#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import pandas as pd
import numpy as np
import json

def extractSubContinent(geoNetwork):
	return json.loads(geoNetwork)['subContinent']

def extractTotals(totals):
	totals = json.loads(totals)
	if ('transactionRevenue' in totals):
		return [int(totals['transactionRevenue']), int(totals['hits'])]
	else:
		return [0, int(totals['hits'])]

def extractOperatingSystem(device):
	return json.loads(device)['operatingSystem']

def extractDeviceData(device):
	source = json.loads(device)

	return [source['operatingSystem'], source['deviceCategory']]

def extractTrafficSourceData(trafficSource):
	source = json.loads(trafficSource)

	return [source['campaign'], source['source'], source['medium']]

def aggregateDataByVisitorId(data, aggregated, columns):
	for column in columns:
		print('Aggregating ' + column + '.. ', end='', flush=True)

		groupingCounts = data.groupby(['fullVisitorId', column]).size().reset_index().rename(columns = {0: 'count'})
		groupingCounts = groupingCounts.pivot(index='fullVisitorId', columns=column, values='count').fillna(0)
		groupingCounts.columns = groupingCounts.columns.astype('object').values
		groupingCounts = groupingCounts.astype('int64')

		aggregated = pd.concat([aggregated, groupingCounts], axis=1, sort=False)
		print('Done.')

	return aggregated

def clusterTrafficSources(source):
	if 'google' in source or 'lmgtfy' in source or 'golang' in source:
		return 'goog'
	elif 'youtube' in source:
		return 'youtube'
	elif 'facebook' in source:
		return 'fb'
	elif 'baidu' in source:
		return 'baidu'
	elif 'reddit' in source:
		return 'reddit'
	elif 'qitta' in source:
		return 'qitta'
	elif 'quora' in source:
		return 'quora'
	elif 'bing' in source:
		return 'bing'
	elif 't.co' == source or 'twitter' in source:
		return 'twitter'
	elif 'yahoo' in source:
		return 'yahoo'
	elif '(direct)' == source:
		return 'direct source'
	elif '(not set)' == source:
		return 'source not set'
	elif 'Partners' == source or 'dfa' == source:
		return source
	else:
		return 'other source'

def clusterTrafficCampaigns(campaign):
	if '(not set)' == campaign:
		return 'other'
	elif 'Data Share' in campaign:
		return 'data share'
	elif 'AW' in campaign:
		return 'AW'
	else:
		return 'other'

def clusterSubContinents(subContinent):
	if subContinent == 'Southern Africa' \
			or subContinent == 'Eastern Africa' \
			or subContinent == '(not set)' \
			or subContinent == 'Central Asia' \
			or subContinent == 'Middle Africa' \
			or subContinent == 'Melanesia' \
			or subContinent == 'Micronesian Region' \
			or subContinent == 'Polynesia':
		return 'otherContinent'
	else:
		return subContinent

def clusterOperatingSystems(operatingSystem):
	if operatingSystem == 'Windows' \
			or operatingSystem == 'Macintosh' \
			or operatingSystem == 'Android' \
			or operatingSystem == 'iOS' \
			or operatingSystem == 'Linux' \
			or operatingSystem == 'Chrome OS':
		return operatingSystem
	else:
		return 'otherOS'

print('Loading data.. ', end='', flush=True)
data = pd.read_csv(
	'../data/test.csv',
	date_parser = lambda x: pd.datetime.strptime(x, '%Y%m%d'),
	parse_dates=['date'],
	dtype={
		'channelGrouping': 'category',
		#'date': 'datetime64', inferred using parse_dates
		'device': object,
		'fullVisitorId': object,
		'geoNetwork': object,				#Might use at some point
		'sessionId': object,
		'socialEngagementType': 'category',
		'totals': object,
		'trafficSource': object,
		'visitId': 'int64',
		'visitNumber': 'int64',
		'visitStartTime': 'int64'
	}
)
print('Done.')


print('Removing unused fields.. ', end='', flush=True)
data = data.drop(columns=['socialEngagementType'])
data = data.drop(columns=['sessionId'])
print('Done.')


#Prepare the aggregated dataframe
aggregated = pd.DataFrame({'fullVisitorId': data['fullVisitorId'].unique()})
aggregated = aggregated.set_index('fullVisitorId')


print('Extracting visit totals from JSON.. ', end='', flush=True)
#Extract transactionRevenue from totals, drop that JSON
data['transactionRevenue'], data['hits'] = zip(*data['totals'].map(extractTotals))
data = data.drop(columns=['totals'])
print('Done')


print('Extracting subContinent from JSON.. ', end='', flush=True)
#Extract transactionRevenue from totals, drop that JSON
data['subContinent'] = data['geoNetwork'].map(extractSubContinent)
data = data.drop(columns=['geoNetwork'])
#Group together subcontinent values
data['subContinent'] = data['subContinent'].apply(clusterSubContinents)
print('Done')


print('Extracting device data from JSON.. ', end='', flush=True)
#Extract transactionRevenue from totals, drop that JSON
data['operatingSystem'], data['deviceCategory'] = \
	zip(*data['device'].map(extractDeviceData))
data = data.drop(columns=['device'])
#Group together operatingSystem values
data['operatingSystem'] = data['operatingSystem'].apply(clusterOperatingSystems)
print('Done')


print('Extracting trafficSource from JSON.. ', end='', flush=True)
#Extract transactionRevenue from totals, drop that JSON
data['campaign'], data['source'], data['medium'] = \
	zip(*data['trafficSource'].map(extractTrafficSourceData))
data = data.drop(columns=['trafficSource'])
#Group together traffic column values
data['campaign'] = data['campaign'].apply(clusterTrafficCampaigns)
data['source'] = data['source'].apply(clusterTrafficSources)
data.loc[(data['medium'] == '(not set)') | (data['medium'] == '(none)'), 'medium'] = 'no medium'
print('Done')

print('Aggregating total transactions.. ', end='', flush=True)
aggregated['totalTransactionRevenue'] = np.log1p(data.groupby('fullVisitorId')['transactionRevenue'].sum())
aggregated['nonZeroRevenue'] = aggregated['totalTransactionRevenue'].map(lambda revenue: 1 if revenue > 0 else 0)
print('Done.')


columns = [
	'channelGrouping',
	'deviceCategory',
	'operatingSystem',
	'subContinent',
	'campaign',
	'source',
	'medium'
]
aggregated = aggregateDataByVisitorId(data, aggregated, columns)


#Compute the number of hits
groupingCounts = data.pivot_table(index='fullVisitorId', values='hits', aggfunc=sum).fillna(0)
groupingCounts = pd.DataFrame(groupingCounts)
aggregated = pd.concat([aggregated, groupingCounts], axis=1, sort=False)


#Compute the number of visits
print('Aggregating visit count.. ', end='', flush=True)
aggregated['visitCount'] = data['fullVisitorId'].value_counts()
print('Done.')


#Write the results to file
print('Writing aggregated data to file.. ', end='', flush=True)
aggregated = aggregated.reset_index()
aggregated = aggregated.rename(columns={"index": "fullVisitorId"})
aggregated.to_csv('../data/testAggregate.csv', index=False)
print('Done.')

