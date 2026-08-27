import unittest
from typing import ClassVar

import pytest

from src.schemas.dag import LinearizedDAG
from src.utils import list_to_dict_by_task, load_yaml


async def run_di(prompt: str):
    test_dag = LinearizedDAG(prompt)
    return await test_dag.launch_test()


class TestExample(unittest.TestCase):
    passed_tests_counter = 0
    config: ClassVar[dict]
    task_to_gt: ClassVar[dict]

    @classmethod
    def setUpClass(cls):
        cls.config = load_yaml("tests/testcases/example.yaml")
        cls.task_to_gt = list_to_dict_by_task(cls.config["TMC-list"])

    @pytest.mark.skip(reason="Заглушка без проверок: тело теста пустое")
    def test_larger_than(self):
        pass

    @pytest.mark.skip(reason="Заглушка без проверок: сравнивает константу 0 с 10")
    def test_distance(self):
        distance = 0

        self.assertLess(distance, 10)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExample)
    runner = unittest.TextTestRunner()
    runner.run(suite)
    return TestExample.passed_tests_counter


if __name__ == "__main__":
    passed_tests = run_tests()
    print(f"\nTotal passed tests: {passed_tests}")
