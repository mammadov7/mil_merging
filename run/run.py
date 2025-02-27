import numpy as np 
size=20


data= [ 'std_barlow_res', 'bracs_dino_res']
init_types= [ 'uniform','xavier', 'orthogonal', 'switch']

lrs=[ 0.01, 0.01, 0.01,  0.01, 0.01,  0.01, 0.01, 0.01, 0.01,  0.01,]  


n=[ 384, 512, 384, 512, 
    384, 512, 384, 512,
    384, 512, 384, 512,
    512, 512
]

nb_classes=[1,3]

q_n=[128,32,64,128]
average = ['True','False']
dropout_patch =[0.2,0.0,0.3,0.4]
dropout_node = [0.2,0.0,0.3,0.3]
non_linearity = [1,1,0,0]

text_file = open('run.sh','r')
lines = text_file.readlines()

for i in range(size):
    line_list = lines.copy()
    line_list[9]=line_list[9].replace('?',str(n[1]))
    line_list[10]=line_list[10].replace('?',str(nb_classes[i//10]))
    line_list[11]=line_list[11].replace('?',str('{:f}'.format(lrs[0])))
    line_list[12]=line_list[12].replace('?',str('{:f}'.format(0.00001)))
    line_list[13]=line_list[13].replace('?',str('{:f}'.format(dropout_node[1])))
    line_list[14]=line_list[14].replace('?',str('{:f}'.format(dropout_patch[1])))
    line_list[15]=line_list[15].replace('?',str(non_linearity[-1]))
    line_list[16]=line_list[16].replace('?',str(128))
    line_list[17]=line_list[17].replace('?',str(average[0]))
    line_list[18]=line_list[18].replace('?',str(data[i//10]))
    line_list[20]=line_list[20].replace('?',str(i%10+1))
    line_list[21]=line_list[21].replace('?',str((10)))
    line_list[22]=line_list[22].replace('?',str(init_types[2]))
    line_list[23]=line_list[23].replace('?',str(i+1))
    out_file = open('run'+str(i+1)+'.sh','w')
    out_file.writelines(line_list)
