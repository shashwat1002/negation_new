import datasets
import json
from transformers import RobertaForMaskedLM, RobertaTokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer
from icecream import ic
import torch
from torch.nn import Softmax

TRAIN_LAMA = "../initial_experiments/lama_train.txt"
TEST_LAMA = "../initial_experiments/lama_test.txt"
DEV_LAMA = "../initial_experiments/lama_dev.txt"

DEVICE = "cpu"

def exact_string_match(full_string, sub_string):
    """
    Takes the full_string, and the substring, and returns the indices of the substring in the full string
    :param full_string: string
    :param sub_string: string
    """
    start = full_string.find(sub_string)
    end = start + len(sub_string)
    return start, end

def replace_substring_with_mask(full_string, sub_string, mask_token):
    """
    Takes the full_string, and the substring, and returns the full string with the substring replaced with the mask token
    :param full_string: string
    :param sub_string: string
    :param mask_token: string
    """
    start, end = exact_string_match(full_string, sub_string)
    return full_string[:start] + mask_token + full_string[end:]

def send_mask_logits(model, tokenizer, sentence):
    inputs_tran = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    # print(inputs_tran)


    with torch.no_grad():
        logits = model(**inputs_tran).logits


    mask_token_index = (inputs_tran.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    # print(mask_token_index)

    return logits[:, mask_token_index, :]

def top_k(k, logits_tensor):
    # print(logits_tensor.size())
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
    # print(top_k_indices_list)
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
                    "subject_hidden": datasets.Value("string"),
                }
            )
        )


    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.config_name = kwargs.get("config_name")
        self.model_name = kwargs.get("model_name")
        self.mask_token = kwargs.get("mask_token")

        # print(self.model_name)

        if kwargs.get("model_name") is None:
            self.model_name = "roberta-base"

        if self.mask_token is None:
            if self.model_name == "roberta-base":
                self.mask_token = "<mask>"
            else:
                self.mask_token = "[MASK]"


        if kwargs.get("model") is None:
            self.model = AutoModelWithLMHead.from_pretrained(self.model_name).to(DEVICE).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            self.model = kwargs.get("model")
            self.tokenizer = kwargs.get("tokenizer")






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
                    dict_rep["masked_sentences"][0].replace("[MASK]", self.mask_token).replace(" .", "."),
                    dict_rep["masked_negations"][0].replace("[MASK]", self.mask_token).replace(" .", "."),
                )

                subject = dict_rep["sub_label"]
                masked_non_negated = pair_of_pair[0]
                label = compare_n_sentences(self.model, self.tokenizer, pair_of_pair)

                subject_hidden = replace_substring_with_mask(masked_non_negated, subject, self.mask_token)

                yield (id_), {
                    "subject": subject,
                    "masked_non_negated": masked_non_negated,
                    "subject_hidden": subject_hidden,
                    "label": label,
                    "id_": id_
                }


