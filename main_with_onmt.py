# pretrain
# wget https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
# tar xf toy-ende.tar.gz
# cd toy-ende

# head -n 3 toy-ende/src-train.txt

import os
import numpy as np
from random import choice, randrange


# onmt_build_vocab -config toy_en_de.yaml - n_sample 10000
# onmt_train -config toy_en_de.yaml

def sample(file_name, file_revert, numberOfStrings, min_length=3, max_length=15):
    with open(file_name, 'w') as f,\
            open(file_revert, 'w') as f_r:
        for i in range(numberOfStrings):
            random_length = randrange(min_length, max_length)  # Pick a random length
            random_char_list = [choice(characters[:-1]) for _ in range(random_length)]  # Pick random chars
            random_string = ' '.join(random_char_list)
            random_revert = ''.join([x for x in random_string[::-1]])
            f.write(random_string + '\n')
            f_r.write(random_revert + '\n')
            print(random_string)
            print(random_revert)


# generate data for the revert toy
if __name__ == '__main__':
    characters = list("abcd")
    sample('toy-revert/src-train.txt', 'toy-revert/trg-train.txt', 10000)
    sample('toy-revert/src-val.txt', 'toy-revert/trg-val.txt', 1000)
    sample('toy-revert/src-test.txt', 'toy-revert/trg-test.txt', 1000)
