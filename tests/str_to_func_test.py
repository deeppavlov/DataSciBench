from src.utils import create_function_from_string

func_str = """
def my_dynamic_function(x, y):
    return x - y
"""


def test_create_function_from_string():
    func_name, my_func = create_function_from_string(func_str)

    assert func_name == "my_dynamic_function"
    assert my_func(10, 3) == 7
