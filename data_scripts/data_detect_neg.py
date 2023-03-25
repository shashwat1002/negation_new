import datasets
import json

TRAIN_LAMA = "../initial_experiments/lama_train.txt"
TEST_LAMA = "../initial_experiments/lama_test.txt"
DEV_LAMA = "../initial_experiments/lama_dev.txt"


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
                mask_replace = dict_rep["obj_label"]
                pair_of_pair = (
                    (dict_rep["masked_sentences"][0].replace("[MASK]", mask_replace).replace(" .", "."), 0,),
                    (dict_rep["masked_negations"][0].replace("[MASK]", mask_replace).replace(" .", "."), 1,),
                )

                for i, pair in enumerate(pair_of_pair):
                    id_ += 1
                    yield (id_), {
                        "content": pair[0],
                        "label": pair[1],
                        "id_": id_
                    }
