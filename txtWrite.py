from getFrame import get_frame

path = '/home/yi/Desktop/AFL/dataset/'
file = open(path+'t.txt', 'r')
line = file.readline()
newfile = open(path+'new_train.txt', 'w')

while line != "":
    line = line.replace('\n', '')
    d = path + line
    frames = get_frame(d)

    if 'kick' in line:
        cls = 0
    #elif 'cmark' in line or 'con' in line:
    elif 'pass' in line:
        cls = 2
    else:
        cls = 1

    newline = line + ' ' + str(frames) + ' '+str(cls) +'\n'
    newfile.write(newline)

    line = file.readline()
file.close()
newfile.close()