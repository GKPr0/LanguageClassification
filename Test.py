import json
import math
from dataclasses import dataclass
import fasttext

from Utils import load_model, load_test_data, create_per_line_char_ngram


def test(test_data, eval_fcn):
    total = len(test_data)
    correct = 0
    faulty_tests = []
    for test in test_data:
        expected_lng, sentences = test

        recognized_lng = eval_fcn(sentences)

        if expected_lng == recognized_lng:
            correct += 1
        else:
            faulty_tests.append({
                "Expected": expected_lng,
                "Recognized": recognized_lng,
                "Content": sentences
            })

    report = {
        "Total": total,
        "Correct": correct,
        "Accuracy": (correct / total) * 100,
        "FaultyTests": faulty_tests
    }

    return report


def my_model_evaluate(cz_model, sk_model):
    def calculate_score(sentences: [str], model):
        ngram_order = len(list(model.keys())[0])
        ngrams = create_per_line_char_ngram(sentences, ngram_order)

        # Pro větší numerickou stabilitu převedeno na log a sčítání
        score = 0
        for ngram in ngrams:
            score += math.log(model[ngram])

        return score

    def eval(sentences: [str]):
        cs_score = calculate_score(sentences, cz_model)
        sk_score = calculate_score(sentences, sk_model)

        recognized_lng = "CS" if cs_score > sk_score else "SK"

        return recognized_lng

    return eval


def fast_text_evaluate(model):
    def eval(sentences: [str]):
        result = model.predict(sentences)[0]
        try:
            return str(result[0][0]).replace("__label__", "")
        except:
            return "-"

    return eval


@dataclass
class TestScenario:
    root: str
    languages: [str]
    models: {}
    output: str


if __name__ == "__main__":

    scenarios = [
        TestScenario(root="TestData", languages=["CS", "SK"],
                     models={"CS": "Models/CS/cs_3_model.pkl",
                             "SK": "Models/SK/sk_3_model.pkl"},
                     output="Results/3_model_result.json"),
        TestScenario(root="TestData", languages=["CS_no", "SK_no"],
                     models={"CS": "Models/CS/cs_3_model.pkl",
                             "SK": "Models/SK/sk_3_model.pkl"},
                     output="Results/3_model_text_no_diac_result.json"),
        TestScenario(root="TestData", languages=["CS", "SK"],
                     models={"CS": "Models/CS/cs_3_no_diac_model.pkl",
                             "SK": "Models/SK/sk_3_no_diac_model.pkl"},
                     output="Results/3_model_no_diac_result.json"),
        TestScenario(root="TestData", languages=["CS_no", "SK_no"],
                     models={"CS": "Models/CS/cs_3_no_diac_model.pkl",
                             "SK": "Models/SK/sk_3_no_diac_model.pkl"},
                     output="Results/3_no_diac_model_text_no_diac_result.json"),
        TestScenario(root="TestData", languages=["CS", "SK"],
                     models={"Fast": "Models/3_fasttext_model.bin"},
                     output="Results/3_fasttext_results.json"),
        TestScenario(root="TestData", languages=["CS_no", "SK_no"],
                     models={"Fast": "Models/3_fasttext_model.bin"},
                     output="Results/3_fasttext_text_no_diac_result.json"),
        TestScenario(root="TestData", languages=["CS", "SK"],
                     models={"Fast": "Models/3_fasttext_no_diac_model.bin"},
                     output="Results/3_fasttext_no_diac_results.json"),
        TestScenario(root="TestData", languages=["CS_no", "SK_no"],
                     models={"Fast": "Models/3_fasttext_no_diac_model.bin"},
                     output="Results/3_fasttext_no_diac_text_no_diac_result.json")
    ]

    for scenario in scenarios:
        test_data = load_test_data(root=scenario.root, languages=scenario.languages)

        isFastText = len(scenario.models) == 1
        if isFastText:
            eval_fcn = fast_text_evaluate(fasttext.load_model(scenario.models["Fast"]))
        else:
            cz_model_path = scenario.models
            eval_fcn = my_model_evaluate(cz_model=load_model(scenario.models["CS"]),
                                         sk_model=load_model(scenario.models["SK"]))

        results = test(test_data, eval_fcn=eval_fcn)

        report = {
            "Source": {
                "Root": scenario.root,
                "Languages": scenario.languages,
                "Models": scenario.models,
            },
            "Results": results
        }

        with open(scenario.output, "w", encoding="utf8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
