import copy
import json
import logging
import os

import torch
from torch.utils.data import TensorDataset
from keras.utils import pad_sequences
from utils import get_intent_labels


logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, intent_label_id):
        self.input_ids = input_ids
        self.intent_label_id = intent_label_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)

        self.input_text_file = "seq.in"
        self.intent_label_file = "label"

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, intents, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, intent) in enumerate(zip(texts, intents)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. intent
            intent_label = (
                self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            )
            
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.token_level, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(
            texts=self._read_file(os.path.join(data_path, self.input_text_file)),
            intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
            set_type=mode,
        )


processors = {"syllable-level": JointProcessor, "word-level": JointProcessor}


def convert_examples_to_features(
    examples,
    max_seq_len,
    tokenizer,
):

    features = []
    for (ex_index, example) in enumerate(examples):
        sequence = tokenizer.texts_to_sequences([example.words])
        input_ids = pad_sequences(sequence, maxlen=max_seq_len)[0]
        intent_label_id = int(example.intent_label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                intent_label_id=intent_label_id,
            )
        )

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.token_level](args)

    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode, args.token_level, args.max_seq_len
        ),
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(
            examples, args.max_seq_len, tokenizer
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_intent_label_ids
    )
    return dataset
