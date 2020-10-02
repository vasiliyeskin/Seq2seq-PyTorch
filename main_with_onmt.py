# pretrain
# wget https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
# tar xf toy-ende.tar.gz
# cd toy-ende

# head -n 3 toy-ende/src-train.txt

# onmt_build_vocab -config toy_en_de.yaml - n_sample 10000
# onmt_train -config toy_en_de.yaml