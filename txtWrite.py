from getFrame import get_frame

path = '/home/yi/Desktop/AFL/dataset/'
file = open(path+'list.txt', 'r')
line = file.readline()
newfile = open(path+'all.txt', 'w')

while line != "":
    line = line.replace('\n', '')
    d = path + line
    frames = get_frame(d)

    if 'kick' in line:
        cls = 0
    elif 'cmark' in line or 'con' in line:
        cls = 1
    elif 'mark' in line:
        cls = 2
    elif 'pass' in line:
        cls = 3
    elif 'non' in line:
        cls = 4
    else:
        continue

    newline = line + ' ' + str(frames) + ' '+str(cls) +'\n'
    newfile.write(newline)

    line = file.readline()
file.close()
newfile.close()