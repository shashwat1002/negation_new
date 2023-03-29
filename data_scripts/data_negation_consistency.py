import datasets
import json
from transformers import RobertaForMaskedLM, RobertaTokenizer
from icecream import ic
import torch
from torch.nn import Softmax

TRAIN_LAMA = "../initial_experiments/lama_train.txt"
TEST_LAMA = "../initial_experiments/lama_test.txt"
DEV_LAMA = "../initial_experiments/lama_dev.txt"

DEVICE = "cpu"

def send_mask_logits(model, tokenizer, sentence):
    inputs_tran = tokenizer(sentence, return_tensors="pt").to(DEVICE)



    with torch.no_grad():
        logits = model(**inputs_tran).logits


    mask_token_index = (inputs_tran.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]


    return logits[:, mask_token_index, :]

def top_k(k, logits_tensor):

    sorted_logits, indices = torch.sort(logits_tensor, descending=True)

    return sorted_logits[:k], indices[:k]

def compare_n_sentences(model, tokenizer, sentence_list):
    k = 20

    masked_logits_list = [send_mask_logits(model, tokenizer, sentence) for sentence in sentence_list]

    softmax_func = Softmax(dim=-1)
    preds_prob_list = [softmax_func(masked_logits.squeeze()) for masked_logits in masked_logits_list]

    top_k_preds_list = []
    top_k_indices_list = []

    for preds_prob in preds_prob_list:
        top_k_preds, top_k_indices = top_k(k, preds_prob)
        top_k_preds_list.append(top_k_preds)
        top_k_indices_list.append(top_k_indices)


    pred_tokens = [tokenizer.convert_ids_to_tokens(indices.squeeze()) for indices in top_k_indices_list]

    return not(top_k_indices_list[0].squeeze()[0] == top_k_indices_list[1].squeeze()[0])

    # return out_string

class NegationConsistency(datasets.BuilderConfig):

    def __init__(self, features, label_classes=("Not Consistent", "Consistent",), **kwargs):
        super().__init__(kwargs)

        self.features = features
        self.label_classes = label_classes



class NegationConsistency(datasets.GeneratorBasedBuilder):

    """
    label: check if a lama sentence has negation consistency (true, false)
    content: the subject string
    """

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id_": datasets.Value("int32"),
                    "subject": datasets.Value("string"),
                    "label": datasets.Value("int32"),
                    "masked_non_negated": datasets.Value("string"),
                }
            )
        )


    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.config_name = kwargs.get("config_name")

        self.model = RobertaForMaskedLM.from_pretrained('roberta-base').to(DEVICE).eval()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')






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
                id_ += 1
                try:
                    dict_rep = json.loads(line)
                except json.decoder.JSONDecodeError:
                    pass
                mask_replace = "<mask>"
                pair_of_pair = (
                    dict_rep["masked_sentences"][0].replace("[MASK]", mask_replace).replace(" .", "."),
                    dict_rep["masked_negations"][0].replace("[MASK]", mask_replace).replace(" .", "."),
                )

                subject = dict_rep["sub_label"]
                masked_non_negated = pair_of_pair[0]
                label = compare_n_sentences(self.model, self.tokenizer, pair_of_pair)

                yield (id_), {
                    "subject": subject,
                    "masked_non_negated": masked_non_negated,
                    "label": label,
                    "id_": id_
                }


