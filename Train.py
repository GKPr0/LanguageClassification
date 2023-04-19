from dataclasses import dataclass

import fasttext

from Utils import create_smoothed_ngram, save_model


@dataclass
class TrainScenario:
    source: str
    ngram: [int]
    output: str
    fast_text: bool = False
    fast_kwargs: dict = None


if __name__ == "__main__":

    scenarios = [
        TrainScenario(source="TrainData/CS_train.txt", ngram=3, output="Models/CS/cs_3_model.pkl"),
        TrainScenario(source="TrainData/CS_no_diac_train.txt", ngram=3, output="Models/CS/cs_3_no_diac_model.pkl"),
        TrainScenario(source="TrainData/SK_train.txt", ngram=3, output="Models/SK/sk_3_model.pkl"),
        TrainScenario(source="TrainData/SK_no_diac_train.txt", ngram=3, output="Models/SK/sk_3_no_diac_model.pkl"),
        TrainScenario(source="TrainData/fastText_train.txt", ngram=3, output="Models/3_fasttext_model.bin",
                      fast_text=True, fast_kwargs={"epoch": 50, "lr": 0.05}),
        TrainScenario(source="TrainData/fastText_no_diac_train.txt", ngram=3, output="Models/3_fasttext_no_diac_model.bin",
                      fast_text=True, fast_kwargs={"epoch": 50, "lr": 0.05}),
    ]

    for scenario in scenarios:
        if scenario.fast_text:
            model = fasttext.train_supervised(scenario.source, minn=scenario.ngram, maxn=scenario.ngram, **scenario.fast_kwargs)
            model.save_model(scenario.output)
        else:
            with open(scenario.source, encoding="utf8") as f:
                train_data = f.readlines()
            cz_model = create_smoothed_ngram(train_data, ngram_order=scenario.ngram)
            save_model(cz_model, scenario.output)
