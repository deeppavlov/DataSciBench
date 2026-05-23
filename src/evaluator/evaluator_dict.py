from .cr_evaluator import CREvaluator
from .larger_than_evaluator import LargerthanEvaluator
from .within_range_evaluator import WithinRangeEvaluator

TM_2_EVALUATOR = {
    "CR": CREvaluator,
    "test_bool": WithinRangeEvaluator,
    "test_int": LargerthanEvaluator
}