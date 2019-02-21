

a = {'social sciences and society': 0, 'sports and recreation': 1, 'natural sciences': 2, 'language and literature': 3, 'geography and places': 4, 'music': 5, 'media and drama': 6, 'art and architecture': 7, 'warfare': 8, 'engineering and technology': 9, 'video games': 10, 'philosophy and religion': 11, 'agriculture, food and drink': 12, 'history': 13, 'mathematics': 14, 'miscellaneous': 15, 'media and darama': 16, 'unk': 17}

res = dict((v, k) for k,v in a.items())

def read_dataset(filename):
    list = []
    with open(filename, "r") as f:
        for line in f:
            index = int(line.strip())
            list.append(res[index])
    return list

def write_data(my_list,filename):
    with open(filename, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)


dev = read_dataset("dev.txt")
test = read_dataset("test.txt")

write_data(dev, "dev_labels.txt")
write_data(test,"test_labels.txt")



