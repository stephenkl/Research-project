from getFrame import get_frame

path = '/home/yi/Desktop/AFL/test_final/'
file = open(path+'list.txt', 'r')
line = file.readline()
newfile = open(path+'all.txt', 'w')

kick_count = 0
cmark_count = 0
mark_count = 0
pass_count = 0
non_count = 0


while line != "":
    line = line.replace('\n', '')
    d = path + line
    frames = get_frame(d)

    if 'kick' in line:
        cls = 0
        kick_count += 1
    elif 'cmark' in line or 'con' in line:
        cls = 1
        cmark_count+=1
    elif 'mark' in line:
        cls = 2
        mark_count+=1
    elif 'pass' in line:
        cls = 3
        pass_count+=1
    elif 'non' in line:
        cls = 4
        non_count+=1
    else:
        continue

    newline = line + ' ' + str(frames) + ' '+str(cls) +'\n'
    newfile.write(newline)

    line = file.readline()
file.close()
newfile.close()
print('kick:',kick_count)
print('cmark:',cmark_count)
print('mark', mark_count)
print('pass', pass_count)
print('non', non_count)