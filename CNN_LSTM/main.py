import argparse

from data_loader import load_and_cache_examples
from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, tokenizer, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")
    if args.do_eval_dev:
        trainer.load_model()
        trainer.evaluate("dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./PhoATIS", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")

    parser.add_argument(
        "--model_type",
        default="cnn",
        type=str,
        help="Model type selected in the list: [cnn, lstm]",
    )
    parser.add_argument("--tuning_metric", default="loss", type=str, help="Metrics to tune when training")
    parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_eval_dev", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")


    parser.add_argument(
        "--token_level",
        type=str,
        default="word-level",
        help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=50,
        help="Number of unincreased validation step to wait for early stopping",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="Select gpu id")
    parser.add_argument("--max_vocab_size", default=10000, type=int, help="Size of vocabulary")
    parser.add_argument("--embedding_dim", default=400, required=True, type=int, help="Size of embedding")
    parser.add_argument("--hidden_size", default=768, type=int, help="Size of hidden state")

    args = parser.parse_args()

    main(args)
