import datasets
import json
import string
import difflib
import random
import re

SEED = 42

TRAIN_LAMA = "../initial_experiments/lama_train.txt"
TEST_LAMA = "../initial_experiments/lama_test.txt"
DEV_LAMA = "../initial_experiments/lama_dev.txt"


def random_char_control(s1, s2, shuffle):
    '''
    Pass shuffle as a 26 length string where 
    shuffle[i] = transformation for 'a' + i (in lowercase)
    '''

    ret = ""
    for i,s in enumerate(difflib.ndiff(s1, s2)):
        if s[0]==' ': 
            ret += s[-1]
        elif s[0]=='-':
            # raise ValueError(f'{s1}-> {s2} involves deletions and random_char_control cant handle that')
            continue
        elif s[0]=='+':
            c = s[-1]
            # print(s[0], s[-1], i)
            if c.isalpha():
                if c.isupper(): #Assuming: we need to preserve upper-case letters
                    c = shuffle[string.ascii_uppercase.index(c)]
                    c = c.upper()
                elif c.islower():
                    c = shuffle[string.ascii_lowercase.index(c)]
            ret += c
    return s1, ret


class NegDetectionProbeDataset(datasets.BuilderConfig):

    def __init__(self, features, label_classes=("Non Contradiction", "Contradiction",), **kwargs):
        super().__init__(kwargs)

        self.features = features
        self.label_classes = label_classes


class NegDetectionProbeDataset(datasets.GeneratorBasedBuilder):


    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id_": datasets.Value("int32"),
                    "content": datasets.Value("string"),
                    "label": datasets.Value("int32")
                }
            )
        )


    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.config_name = kwargs.get("config_name")
        self.control = 0
        try:
            self.control = kwargs.get("control")
        except KeyError:
            pass

    def _split_generators(self, dl_manager):

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"file_path": TRAIN_LAMA}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"file_path": TEST_LAMA}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"file_path": DEV_LAMA})
        ]

    def _generate_examples(self, file_path):

        with open(file_path, "r") as f:
            id_ = 0
            for id_quintuple, line in enumerate(f):
                try:
                    dict_rep = json.loads(line)
                except json.decoder.JSONDecodeError:
                    pass
                mask_replace = dict_rep["obj_label"]
                pair_of_pair = (
                    (dict_rep["masked_sentences"][0].replace("[MASK]", mask_replace).replace(" .", "."), 0,),
                    (dict_rep["masked_negations"][0].replace("[MASK]", mask_replace).replace(" .", "."), 1,),
                )

                if self.control == 1:
                    # replace negation particles with random gibberish
                    s = string.ascii_lowercase[:26]
                    l = list(s)
                    random.Random(SEED).shuffle(l)
                    cipher = ''.join(l)
                    sentences = list(random_char_control(pair_of_pair[0][0], pair_of_pair[1][0], cipher))

                    pair_of_pair = ((sentences[0], 0), (sentences[1], 1))

                replace_neg_with = "actually"
                if self.control == 2:
                    # replace negation with a specific word
                    sentences = [pair_of_pair[0][0], pair_of_pair[1][0]]
                    not_iter = re.finditer(r"\bnot\b", pair_of_pair[1][0])
                    if not_iter is None:
                        continue
                    else:
                        sentences[1] = re.sub(r"\bnot\b", replace_neg_with, sentences[1])

                    pair_of_pair = ((sentences[0], 0), (sentences[1], 1))



                for i, pair in enumerate(pair_of_pair):
                    id_ += 1
                    yield (id_), {
                        "content": pair[0],
                        "label": pair[1],
                        "id_": id_
                    }
