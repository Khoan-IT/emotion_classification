import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from keras.utils import pad_sequences
from tqdm import tqdm
from utils import MODEL_CLASSES, get_intent_labels, get_slot_labels, init_logger, load_tokenizer


logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, "training_args.bin"))


def load_model(pred_config, args, device, tokenizer):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES[args.model_type](
            args=args, tokenizer=tokenizer, intent_label_lst=get_intent_labels(args)
        )
        model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.bin")))
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except Exception:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    labels = []
    with open(pred_config.input_file.split('/seq')[0] + "/label", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            labels.append(line)

    return lines, labels


def convert_input_file_to_tensor_dataset(
    lines,
    pred_config,
    args,
    tokenizer,
):
    
    all_input_ids = []
    for words in lines:
        sequence = tokenizer.texts_to_sequences([words])
        input_ids = pad_sequences(sequence, maxlen=args.max_seq_len)[0]

        all_input_ids.append(input_ids)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)

    dataset = TensorDataset(all_input_ids)

    return dataset


def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    tokenizer = load_tokenizer(args)
    model = load_model(pred_config, args, device, tokenizer)

    intent_label_lst = get_intent_labels(args)
    lines = read_input_file(pred_config)
    dataset = convert_input_file_to_tensor_dataset(lines[0], pred_config, args, tokenizer)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)
    intent_preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "intent_label_ids": None,
            }

            outputs = model(**inputs)
            _, intent_logits, _ = outputs

            # Intent Prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)


    intent_preds = np.argmax(intent_preds, axis=1)


    # Write to output file
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        for words, label, intent_pred in zip(lines[0], lines[1], intent_preds):
            print("{}\t{}\t{}".format(intent_label_lst[intent_pred], label, " ".join(words)), file=f)

    logger.info("Prediction Done!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="sample_pred_in.txt", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="./atis_model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    predict(pred_config)