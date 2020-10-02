import argparse
import torch
import json
import os

from training import train, evaluate
from models.seq2seq import Seq2Seq
from torch.utils import data
from utils.data_generator import ToyDataset, pad_collate

import numpy as np
from random import choice, randrange
import tqdm


def run():
    USE_CUDA = torch.cuda.is_available()

    config_path = os.path.join("experiments", FLAGS.config)

    if not os.path.exists(config_path):
        raise FileNotFoundError

    with open(config_path, "r") as f:
        config = json.load(f)

    config["gpu"] = torch.cuda.is_available()

    dataset = ToyDataset(5, 15)
    eval_dataset = ToyDataset(5, 15, type='eval')
    BATCHSIZE = 30
    train_loader = data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate, drop_last=True)
    eval_loader = data.DataLoader(eval_dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate,
                                  drop_last=True)
    config["batch_size"] = BATCHSIZE

    # Models
    model = Seq2Seq(config)

    if USE_CUDA:
        model = model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", .001))

    print("=" * 60)
    print(model)
    print("=" * 60)
    for k, v in sorted(config.items(), key=lambda i: i[0]):
        print(" (" + k + ") : " + str(v))
    print()
    print("=" * 60)

    print("\nInitializing weights...")
    for name, param in model.named_parameters():
        if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            torch.nn.init.xavier_normal_(param)

    for epoch in range(FLAGS.epochs):
        run_state = (epoch, FLAGS.epochs, FLAGS.train_size)

        # Train needs to return model and optimizer, otherwise the model keeps restarting from zero at every epoch
        # print("My test: ", model('abcd'))
        model, optimizer = train(model, optimizer, train_loader, run_state)
        evaluate(model, eval_loader)



        # dset = my_sample(eval_dataset)
        # tl2 = data.DataLoader(dset, batch_size=1, shuffle=False, collate_fn=pad_collate,
        #                                drop_last=True)
        # t = tqdm.tqdm(tl2)
        # model.eval()
        # for batch in t:
        #     loss, logits, labels, alignments = model.loss()

        # TODO implement save models function

def my_sample(dtset):
        random_length = randrange(dtset.min_length, dtset.max_length)  # Pick a random length
        random_char_list = [choice(dtset.characters[:-1]) for _ in range(random_length)]  # Pick random chars
        random_string = ''.join(random_char_list)
        a = np.array([dtset.char2int.get(x) for x in random_string])
        b = np.array([dtset.char2int.get(x) for x in random_string[::-1]] + [2]) # Return the random string and its reverse
        x = np.zeros((random_length, dtset.VOCAB_SIZE))

        x[np.arange(random_length), a-3] = 1

        return x, b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--train_size', default=28000, type=int)
    parser.add_argument('--eval_size', default=2600, type=int)
    FLAGS, _ = parser.parse_known_args()
    run()



