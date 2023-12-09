from transformers import AutoModel, AutoTokenizer, AutoConfig, BertTokenizer, AutoModelForMaskedLM
from torch.nn import Module
from torch.utils.data import DataLoader
from icecream import ic
from tqdm import tqdm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from pytorch_lightning import LightningModule
import torchmetrics
import os


BERTNOT_CHECKPOINT_PATH = "checkpoint_bertnot/"

class EncoderWrapper(Module):

    """
    Wrapper on encoder that gives mean-pooled representations of each layer in a list
    """

    def __init__(self, model_name="roberta-base", device="cpu"):
        super().__init__()

        if model_name != "bertnot":
            self.device = device
            self.model_name = model_name
            self.model_obj = AutoModel.from_pretrained(
                model_name).eval()
            self.model_obj.eval()
            self.tokenizer_obj = AutoTokenizer.from_pretrained(model_name)
            self.config_obj = AutoConfig.from_pretrained(model_name)
            self.to(device)
        else:
            # if the model to be loaded is BERTNOT
            # then we load BERTNOT custom model from path, we initialize a BERT tokenizer from a vocab.txt
            # and we initialize a BERT config from a config.json
            self.device = device
            self.model_name = model_name
            model_path = os.path.join(BERTNOT_CHECKPOINT_PATH, "pytorch_model.bin")
            config_path = os.path.join(BERTNOT_CHECKPOINT_PATH, "config.json")
            vocab_path = os.path.join(BERTNOT_CHECKPOINT_PATH, "vocab.txt")
            self.model_config = AutoConfig.from_pretrained(config_path)
            self.model_lm = AutoModelForMaskedLM.from_pretrained(model_path, config=self.model_config).eval()
            self.model_obj = self.model_lm.bert
            self.model_obj.eval()
            self.tokenizer_obj = BertTokenizer(vocab_file=vocab_path, do_lower_case=False, max_len=512)
            self.to(device)

    def forward(self, input_text):

        encoder_ret = self.tokenizer_obj(
            input_text, truncation=True, return_tensors="pt", padding=True).to(self.device)

        encoder_text_ids = encoder_ret.input_ids.to(self.device)
        attention_mask = encoder_ret.attention_mask.to(self.device) # 1 for not pad

        ic(encoder_text_ids.device)
        ic(self.model_obj.device)
        encoder_states = self.model_obj(
            encoder_text_ids, output_hidden_states=True, attention_mask=attention_mask)

        hs_tuple = encoder_states["hidden_states"]

        mean_pooled_all_layers = []

        for layer, hs in enumerate(hs_tuple):
            ic(hs_tuple[layer].size())
            # hs = hs_tuple[layer] # (batch_size x sequence_length x dimension)
            hs_masked = hs * attention_mask[:, :, None] # ideally zeros out the pad associated representations
            ic(hs_masked.size())
            seq_lengths = attention_mask.sum(dim=1) # each line here represents sequence length

            hs_masked_sum = hs_masked.sum(dim=1)
            hs_avg = hs_masked_sum / seq_lengths[:, None]
            mean_pooled_all_layers.append(hs_avg)

        return mean_pooled_all_layers


class AcrossLayerfit(LightningModule):
    """
    Maintains a set of parameters to weight the representations of each encoder layer with
    Takes the model as argument that is supposed to fit these
    """

    def __init__(self, num_layers=13, hidden_size=768, model=None, lr=0.01):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        if model is None:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 300),
                torch.nn.ReLU(),
                torch.nn.Linear(300, 200),
                torch.nn.ReLU(),
                torch.nn.Linear(200, 1),
                torch.nn.Sigmoid()
            )
        else:
            self.model = model

        # to weigh the representations from each layer
        self.learnable_parameters = torch.nn.Parameter(torch.randn(self.num_layers))
        self.softmax = torch.nn.Softmax(dim=0)
        self.criterion = torch.nn.BCELoss()
        self.accuracy_metric = torchmetrics.classification.BinaryAccuracy()

    def forward(self, batch):
        # assuming that batch is a list with representations from each layer
        # batch_tensor = torch.stack(batch, dim=0)
        # batch_tensor_view = batch_tensor.view(batch_tensor.size()[1], batch_tensor.size()[0], -1) # so that batch size is first dimension
        scores = self.softmax(self.learnable_parameters) # scores from the parameters
        weighted_sum = torch.matmul(scores, batch) # weighted sum

        return self.model(weighted_sum)

    def training_step(self, batch, batch_idx):
        batch_inp, y = batch
        y = y.float()
        y_pred = self.forward(batch_inp).squeeze()
        loss = self.criterion(input=y_pred, target=y)

        log_dict = {
            "train_loss": loss
        }

        self.log_dict(log_dict)

        return loss

    def validation_test_step(self, batch, batch_idx):
        batch_inp, y = batch
        y = y.float()
        with torch.no_grad():
            y_pred = self.forward(batch_inp).squeeze()
        loss = self.criterion(input=y_pred, target=y)
        accuracy = self.accuracy_metric(preds=y_pred, target=y)

        return loss, accuracy
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.validation_test_step(batch, batch_idx)

        log_dict = {
            "validation_loss": loss,
            "validation_accuracy": accuracy
        }

        self.log_dict(log_dict)
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.validation_test_step(batch, batch_idx)

        log_dict = {
            "test_loss": loss,
            "test_accuracy": accuracy
        }

        self.log_dict(log_dict)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.lr)




def get_hidden_states_many_examples(model, data, n=100, layer=-1, batch_size=1, query_column="content"):
    """
    Takes a bunch of sequences and runs them through encoder to generate the mean-pooled hidden states.

    This is unbatched and kept inefficient for simplicity

    can be done in batches on a GPU to make it faster
    """
    # setup
    model.eval()
    all_hidden_states, all_labels = [], []
    # all_hidden_states: will have elements for each encoder layer, each element represents the mean-pooled representations for the whole data at that layer

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    # loop
    # for idx in tqdm(range(n)):
    for i, batch in enumerate(dataloader):

        if ((i+1) * batch_size) > n:
            break
        text, true_label = batch[query_column], batch["label"].to(model.device)
        ic(text)
        ic(true_label)


        # get hidden states
        with torch.no_grad():
            outs = model(text)
        # outs: [hidden states]
        ic(outs[0].size())

        # initialize if empty
        if len(all_hidden_states) == 0:
            for i in range(len(outs)):
                all_hidden_states.append([])


        # collect
        for i, hidden_state in enumerate(outs):
            all_hidden_states[i].append(hidden_state)

        all_labels.append(true_label)

    ic(len(all_hidden_states))
    ic(len(all_hidden_states[0]))
    ic(all_hidden_states[0][0].size())
    ic(torch.cat(all_hidden_states[0], dim=0).size())

    all_hidden_states = [torch.cat(all_hidden_states[i], dim=0) for i in range(len(all_hidden_states))]


    return all_hidden_states, torch.cat(all_labels, dim=0)

def run_experiment_across_layers(experiment, train_input, train_labels, test_input, test_labels):
    """
    Runs a probing experiment over representations from all layers of the model.
    The whole thing works on cached embeddings

    experiment: method (train: Tensor, test: Tensor, label_train: Tensor, label_test: Tensor) -> (fit_model, metrics). Each experiment will fit _some_ model on the data and return the model and the results
    train_input: list of 13 elements, each of which is a tensor of size (num_datapoints, embedding_dim)
    train_labels: tensor (num_datapoints, )
    test_input: same format as train_input
    test_labels: same format as train_labels
    """

    list_of_results = []
    list_of_probing_models = []

    for i in range(len(train_input)):
        train_current_layer = train_input[i]
        test_current_layer = test_input[i]

        model, metrics = experiment(train_current_layer, test_current_layer, train_labels, test_labels)

        list_of_results.append(metrics)
        list_of_probing_models.append(model)

    return list_of_probing_models, list_of_results

def probe_experiment(train_input, test_input, train_labels, test_labels, probe_model):
    """
    Gets an initialized probe model and fits it on data and runs some experiments
    expected to be curried and sent as a callback to run_experiment_across_layers
    """

    train_input_numpy = train_input.detach().cpu().numpy()
    test_input_numpy = test_input.detach().cpu().numpy()
    ic(train_labels.size())
    train_labels_numpy = train_labels.detach().cpu().numpy()
    test_labels_numpy = test_labels.detach().cpu().numpy()

    probe_model.fit(train_input_numpy, train_labels_numpy)

    accuracy = probe_model.score(test_input_numpy, test_labels_numpy)

    return probe_model, {"accuracy": accuracy}


def linear_probe_experiment(train_input, test_input, train_labels, test_labels):
    # initialize linear probe and run probe experiment
    lr = LogisticRegression(class_weight="balanced", verbose=1, max_iter=1000)
    return probe_experiment(train_input, test_input, train_labels, test_labels, lr)


def mlp_probe_experiment(train_input, test_input, train_labels, test_labels):
    # initialize an mlp probe and run probe experiment
    mlp = MLPClassifier(random_state=1, max_iter=1000, verbose=True, hidden_layer_sizes=(300, 100))
    print("here")
    return probe_experiment(train_input, test_input, train_labels, test_labels, mlp)


def generate_classification_report_all_layers(input_all_layers, labels, models):
    """
    input_all_layers is a list of 13 layers
    labels is a tensor
    """
    labels = labels.detach().cpu().numpy()
    preds_for_all_layers = [models[i].predict(input_for_layer.detach().cpu().numpy()) for i, input_for_layer in enumerate(input_all_layers)]
    classification_reports = [classification_report(y_true=labels, y_pred=pred, output_dict=True) for pred in preds_for_all_layers]
    return classification_reports



def collect_all_precisions_recalls(classification_report_list):
    for_class_zero = [report['0'] for report in classification_report_list]
    for_class_one = [report['1'] for report in classification_report_list]

    return for_class_zero, for_class_one
