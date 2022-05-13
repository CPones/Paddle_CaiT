f1 = open(r'data/LSVRC2012/3train_list_shuffle_cut2.txt','rb')
f2= open(r'data/LSVRC2012/train_list.txt','wb+')
# 生成lite_train
i=0
while True:
    line = f1.readline()
    i+=1
    f2.write(line)
    if i>500:
        break
f1.close()
f2.close()


f1 = open(r'data/LSVRC2012/3valid_list_shuffle_cut2.txt','rb')
f2= open(r'data/LSVRC2012/val_list.txt','wb+')
# 生成lite_val
i=0
while True:
    line = f1.readline()
    i+=1
    f2.write(line)
    if i>500:
        break
f1.close()
f2.close()
print("Sucessfilly genernate lite dataset!")