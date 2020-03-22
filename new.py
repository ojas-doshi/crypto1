from numpy import genfromtxt

symbol_list= ['LTC']
VH="1,0,0,0,0,0"
H="0,1,0,0,0,0"
NC="0,0,1,0,0,0"
L="0,0,0,1,0,0"
VL="0,0,0,0,1,0"
NA="0,0,0,0,0,1"

HIGH="1,0,0"
LOW="0,1,0"
NOCHANGE="0,0,1" 

def getMovement(perc):
    if(perc<-0.1): return VL
    if(perc<-.05): return L
    if(perc>.05): return H
    if(perc>0.1): return VH
    return NC
dataset = genfromtxt('LTC.csv',delimiter=',')
x = dataset[:,2:6]
f = open("LTC1.csv","w+")
lastopen = None
lastclose = None

beta=0.4
prevvalue=None
for i in range(0,len(x)-30):
    curropen = None
    currclose = None
    open = x[i][0]
    high = x[i][1]
    low  = x[i][2]
    close = x[i][3]
    str1 = str(open)
    for j in range(1,30):
        str1 +=","
        v=x[i+j][0]

        if prevvalue is None:
            ov=v
        else:
            ov=v*beta+prevvalue*(1-beta)
        str1+= str(ov)
        prevvalue=ov
    str1 += ","
    if curropen==None:
        curropen  = open
        currclose = close
    f.write(str1)
    if(lastopen!=None and lastclose!=None):
        # print(lastclose,lastopen)
        # perc=((lastclose-lastopen)/lastopen)*100
        # print(perc)
        # f.write(getMovement(perc))
        if lastopen>lastclose:
            f.write(HIGH)
        elif lastopen<lastclose:
            f.write(LOW)
        else:
            f.write(NOCHANGE)
    else:
        f.write(NOCHANGE)
    f.write("\n")
    lastopen=curropen
    lastclose=currclose
f.close()
