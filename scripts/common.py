"""
Файл, содержащий описания морфологических категорий для разных языков
и функции работы с ними
"""

from itertools import product

# коды языков
LANGUAGE_CODES = ["RU", "LA"]

# категории для отдельных языков
# русский
ru_cases = ['им', 'род', 'дат', 'вин', 'тв', 'пр']
ru_numbers = ['ед', 'мн']
# латинский
la_cases = ['ном', 'ген', 'дат', 'акк', 'абл', 'вок']
la_numbers = ['ед', 'мн']

def get_categories_marks(language):
    """
    Возвращает список категорий в зависимости от языка
    """
    if language not in LANGUAGE_CODES:
        raise ValueError("Only {0} codes are supported.".format(", ".join(LANGUAGE_CODES)))
    if language == "RU":
        marks = list(map(tuple, product(ru_cases, ru_numbers)))
    elif language == "LA":
        marks = list(map(tuple, product(la_cases, la_numbers)))
    return marks
