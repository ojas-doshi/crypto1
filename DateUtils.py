from datetime import datetime
from datetime import timedelta
import math
class DataUtils:
	def getDates(this,d):
		#15 minutes breakpoint
		dc15=datetime(d.year,d.month,d.day,d.hour,math.floor(d.minute/15)*15,0,0)
		dl15=dc15-timedelta(minutes=15)
		#current/last hour
		dch=datetime(d.year,d.month,d.day,d.hour,0,0,0)
		dlh=dch-timedelta(hours=1)
		# current / last 4 Hours
		dc4h = datetime(d.year,d.month,d.day,math.floor(d.hour/4)*4,0,0,0)
		dl4h = dc4h  - timedelta(hours=4)
		# current / last 1 day
		dcd = datetime(d.year,d.month,d.day,0,0,0,0)
		dld = dcd - timedelta(days=1)
		# current / last 1 month
		dc1m = datetime(d.year,d.month,1,0,0,0,0)
		if d.month ==1:
			dl1m =  datetime(d.year,12,1,0,0,0,0)
		else:
			dl1m =  datetime(d.year,d.month-1,1,0,0,0,0)
	
		# current / last 6 months
		dc6m = datetime(d.year,7,1,0,0,0,0)
		dl6m = datetime(d.year,1,1,0,0,0,0)
		
		x =[(dc15,d),(dl15,dc15),(dch,d),(dlh,dch),(dc4h,d),(dl4h,dc4h),(dcd,d),(dld,dcd),(dc1m,d),(dl1m,dc1m - timedelta(days=1)),(dc6m,d),(dl6m,dc6m- timedelta(days=1))]
		
		return (x)
du=DataUtils()
print(du.getDates(datetime.now()))