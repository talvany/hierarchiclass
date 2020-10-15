import pytest
from codebase.test.test_case import SENTENCE_LISTS_CASES
from codebase.model.predictor_single_model import ModelPredictorSingleModel


class TestPrediction(object):

    @pytest.fixture()
    def predictor(self):
        model_folder = "models/out/balanced"
        predictor = ModelPredictorSingleModel(model_folder)
        return predictor

    @pytest.mark.parametrize("sentence, expected", SENTENCE_LISTS_CASES)
    def test_prediction(self, sentence, expected, predictor):
        pred = predictor.predict([sentence])
        assert pred[0] == expected
