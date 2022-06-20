import sys, os

dir_name = str(sys.argv[1])
files = os.listdir(dir_name)

def getmass(l):
    return float(l.split()[2])

data = {}
for file in files:
    info = open(file).readlines()
    a = float(info[1][4:13])
    m = list(map(getmass, info[16:]))
    data[a] = m
    
stdata = str(data)
save_name = 'masses/' + dir_name.split('rockstar_')[1] + '.txt'
print(save_name)
#f = open(save_name,'w')
#f.write(stdata )
#f.close()