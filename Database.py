import mysql.connector
import math
from datetime import datetime
import json
import requests

class Database:

	def __init__(this):
		this.mydb = mysql.connector.connect(host = "127.0.0.1", user = "root", passwd = "root",database = "crypto")
		
	def getValues(this,symbol,fromtime, totime):
		ft=math.floor(datetime.timestamp(fromtime))
		tt=math.floor(datetime.timestamp(totime))

		sql = "SELECT max(high) as max , min(low) as min,avg(close),avg(open) FROM price where symbol='"+symbol+"' and fromtimes>='"+str(ft)+"' and totimes<='"+str(tt)+"'"
		mycursor = this.mydb.cursor(prepared=True)
		mycursor.execute(sql)
		temp = mycursor.fetchall()
		if(len(temp)==0):
			return null
		mycursor.close()
		return temp[0][0], temp[0][1], temp[0][2], temp[0][3]
	


	def close(this):
		this.mydb.close()