from options import read_options

args = read_options()

id_rels = {}

for key, value in args['relation_vocab'].items():
    id_rels[value] = key

print type(id_rels.keys()[0])