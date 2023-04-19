from datasets import load_dataset
import re


def save_to_file(dataset, count, target, prefix=None):
    with open(target, "w", encoding="utf8") as f:
        for row in dataset.take(count):
            text = str(row["text"])
            text = re.sub(r"\s", " ", text)
            if prefix:
                f.write(f"{prefix}; {text}\n")
            else:
                f.write(f"{text}\n")


if __name__ == "__main__":

    train_size = 10000
    test_size = 1000

    cz_dataset = load_dataset("oscar", "unshuffled_deduplicated_cs", split="train", streaming=True)
    # save_to_file(cz_dataset, train_size, f"TrainData/CS/cs_oscar_{train_size}.txt")
    cz_dataset = cz_dataset.skip(train_size)
    save_to_file(cz_dataset, test_size, f"TestData/cs_test_oscar_{test_size}.txt", prefix="CS")

    sk_dataset = load_dataset('oscar', "unshuffled_deduplicated_sk", split='train', streaming=True)
    # save_to_file(sk_dataset, train_size, f"TrainData/SK/sk_oscar_{train_size}.txt")
    sk_dataset = sk_dataset.skip(train_size)
    save_to_file(sk_dataset, test_size, f"TestData/sk_test_oscar_{test_size}.txt", prefix="SK")
