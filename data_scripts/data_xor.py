import datasets
import json

TRAIN_LAMA = "../initial_experiments/lama_train.txt"
TEST_LAMA = "../initial_experiments/lama_test.txt"
DEV_LAMA = "../initial_experiments/lama_dev.txt"

sep_token = "</s>"
cls_token = "<s>"
pad_token = "<pad>"

def transform_lama_to_experiment_format(dictionary, control_task=0, add_sep=True):

    """
    control_task: takes one of three integers
        - 0: no control task, OG experiment
        - 1: control task of replacing negation particles with gibberish and running trained probes
        - 2: Hewitt based control task (elaborated later)

    The Experiment one generated mappings like so:
        - AA -> 0
        - AA' -> 1
        - A'A -> 1
        - A'A' -> 0

    This function transforms it like so when control_task=2:
        - AA -> 0
        - AA' -> 0
        - A'A -> 1
        - A'A' -> 1

    The purpose is a control task
    """

    sentence1 = dictionary["masked_sentences"]
    sentence2 = dictionary["masked_negations"]
    mask_replace = dictionary["obj_label"]

    sentence1 = sentence1[0].replace("[MASK]", mask_replace).replace(" .", ".")
    sentence2 = sentence2[0].replace("[MASK]", mask_replace).replace(" .", ".")

    sentences = [sentence1, sentence2]

    if control_task == 1:
        # replace negation particles with random gibberish
        s = string.ascii_lowercase[:26]
        l = list(s)
        random.Random(shuffler.SEED).shuffle(l)
        cipher = ''.join(l)
        sentences = list(shuffler.random_char_control(sentence1, sentence2, cipher))


    combinations = []

    for i in range(2):
        for j in range(2):
            if add_sep:
                sentence = sentences[i] + f" {sep_token} " + sentences[j]
            else:
                sentence = sentences[i] + f" " + sentences[j]
            outcome = -1

            if control_task == 0:
                if i == j:
                    outcome = 0
                else:
                    outcome = 1
            elif control_task == 2:
                if i == 0:
                    outcome = 0
                else:
                    outcome = 1

            combinations.append((sentence, outcome))

    return combinations

class NegXORDatasetConfig(datasets.BuilderConfig):

    def __init__(self, features, label_classes=("Non Contradiction", "Contradiction",), **kwargs):
        super().__init__(kwargs)

        self.features = features
        self.label_classes = label_classes

class NegXORDataset(datasets.GeneratorBasedBuilder):


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
        self.add_sep = bool(kwargs.get("add_sep"))
        self.config_name = kwargs.get("config_name")

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

                quintuple = transform_lama_to_experiment_format(dict_rep, add_sep=self.add_sep)

                for i, pair in enumerate(quintuple):
                    id_ += 1
                    yield (id_), {
                        "content": pair[0],
                        "label": pair[1],
                        "id_": id_
                    }

# def main():



# if __name__ == "__main__":
#     main()