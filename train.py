from pyfinancialdata import get,get_multi_year
data = get_multi_year([2010,2011,2012], provider='histdata', instrument='SPXUSD')#get(provider='histdata', instrument='SPXUSD', year=2017)

print(data)
