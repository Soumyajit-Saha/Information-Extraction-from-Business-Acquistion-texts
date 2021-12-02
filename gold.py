from os import listdir
from os.path import isfile, join

gold_file=open('gold.templates',"r")
gold=gold_file.read()

gold_file.close()

gold_list=gold.split('\n\n')

file_name=[]

for keys in gold_list:
    if keys != '':
        lines=keys.split('\n')
        print(lines[0])
        name=lines[0].split(': ')[1]
        file_name.append(name)

c=0
for name in file_name:
    txt_file = open('./gold/' + name + '.key', "w")
    txt_file.write(gold_list[c])
    c+=1
    txt_file.close()