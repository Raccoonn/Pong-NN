

def load_data(filename):
    """
    Load in training data.
    """
    f = open(filename)
    rawData    = [[float(i) for i in line.split()] for line in f.read().splitlines()]
    uniqueData = [list(x) for x in set(tuple(x) for x in rawData)]
    print('\n\nSorting Complete\nData Overview:  rawData :', len(rawData), '   uniqueData :', len(uniqueData))
    f.close()
    return rawData, uniqueData


read_filename = 'Training Data/paddle_data_left.txt'

print('\n\nSorting unique Data')

rawData, uniqueData = load_data(read_filename)


print('\n\nWriting unique data to file.\n')

write_file = open('Training Data/paddle_unique_data.txt', 'a')

for line in uniqueData:
    write_file.write('%f %f %f %f %f\n' % tuple(line))

write_file.close()

input('\n\nData write complete, Press <Return> to exit.')