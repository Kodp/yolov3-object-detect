import os
path="yourpath"
files=os.listdir(path)
j=0
for i ,f in enumerate(files):

    if j%15==0:
        print(i)
    else:
        os.remove(path + "/" + str(i) + ".jpg")
    j += 1


