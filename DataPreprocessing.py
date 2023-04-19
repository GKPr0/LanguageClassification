from dataclasses import dataclass

from Utils import load_train_data


@dataclass
class DataPrepScenario:
    root: str
    pattern: str
    languages: [str]
    output: str
    add_label: bool = False
    label_prefix: str = ""


if __name__ == "__main__":

    scenarios = [
        DataPrepScenario(root="TrainData/CS", pattern="*.txt", languages=["CS"], output="TrainData/CS_train.txt"),
        DataPrepScenario(root="TrainData/CS", pattern="*.txt", languages=["CS_no"], output="TrainData/CS_no_diac_train.txt"),
        DataPrepScenario(root="TrainData/SK", pattern="*.txt", languages=["SK"], output="TrainData/SK_train.txt"),
        DataPrepScenario(root="TrainData/SK", pattern="*.txt", languages=["SK_no"], output="TrainData/SK_no_diac_train.txt"),
        DataPrepScenario(root="TrainData", pattern=r"*/*.txt", languages=["CS", "SK"], output="TrainData/fastText_train.txt",
                         add_label=True, label_prefix="__label__"),
        DataPrepScenario(root="TrainData", pattern=r"*/*.txt", languages=["CS_no", "SK_no"], output="TrainData/fastText_no_diac_train.txt",
                         add_label=True, label_prefix="__label__")
    ]

    for scenario in scenarios:
        train_data = load_train_data(root=scenario.root, pattern=scenario.pattern, languages=scenario.languages)
        with open(scenario.output, "w", encoding="utf8") as f:
            for label, data in train_data.items():
                for line in data:
                    if len(line) < 3:
                        continue

                    if scenario.add_label:
                        f.write(f"{scenario.label_prefix}{label} {line}\n")
                    else:
                        f.write(f"{line}\n")
