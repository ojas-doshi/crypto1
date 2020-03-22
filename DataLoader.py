import mysql.connector
import math
from datetime import datetime
from datetime import timedelta
import json
import requests

class DataLoader:
	def __init__(this):
		time = datetime.now()
		try:
			while(1):
				# if time>=datetime.now():
				mydb = mysql.connector.connect(host = "127.0.0.1", user = "root", passwd = "root",database = "crypto")
				mycursor = mydb.cursor(prepared=True)
				symbol_list= ['ETH','EOS','XRP','BCH','LTC','TRX','ETC','BNB','OKB']
				data_link = "https://min-api.cryptocompare.com/data/histominute?fsym=ETH&tsym=BTC&aggregate=15&limit=1"
				response= requests.get(data_link)
				x = response.json()
				y = x["Data"]
				time_ = y[1]['time']
				for k in range(100):
					for j in range(0,len(symbol_list)):
						data_link = "https://min-api.cryptocompare.com/data/histominute?fsym="+symbol_list[j]+"&tsym=BTC&limit=100&toTs="+str(time_)+"&aggregate=15"
						response= requests.get(data_link)
						x = response.json()
						y = x["Data"]
						if (len(y) == 0):
							break
						for i in range(0,len(y)):
							sql = """insert ignore into price (symbol,fromtimes,totimes,fromtime,totime,fromvolume,tovolume,open,high,low,close,processed) values(%s,%s,%s,from_unixtime(%s),from_unixtime(%s),%s,%s,%s,%s,%s,%s,%s)"""
							data = (symbol_list[j],y[i]['time']-900,y[i]['time'],y[i]['time']-900,y[i]['time'],y[i]['volumefrom'],y[i]['volumeto'],y[i]['open'],y[i]['high'],y[i]['low'],y[i]['close'],'0')
							mycursor.execute(sql,data)
							#print("Added")
							mydb.commit()
					try:
						time_=y[0]['time']
					except:
						break
				time=time+timedelta(minutes=15)
				print(time)
		except:
			pass

	def close(this):
		this.mydb.close()

DataLoader()