from pathlib import Path


# -----------------------------
# Python code examples
# -----------------------------

examples = [
    """
def add(a, b):
    return a + b
""",

    """
def subtract(a, b):
    return a - b
""",

    """
def multiply(a, b):
    return a * b
""",

    """
def divide(a, b):
    if b == 0:
        return None
    return a / b
""",

    """
def square(x):
    return x * x
""",

    """
def cube(x):
    return x * x * x
""",

    """
def is_even(number):
    return number % 2 == 0
""",

    """
def is_odd(number):
    return number % 2 != 0
""",

    """
def absolute_value(number):
    if number < 0:
        return -number
    return number
""",

    """
def maximum(a, b):
    if a > b:
        return a
    return b
""",

    """
def minimum(a, b):
    if a < b:
        return a
    return b
""",

    """
def find_max(numbers):
    return max(numbers)
""",

    """
def find_min(numbers):
    return min(numbers)
""",

    """
def calculate_sum(numbers):
    return sum(numbers)
""",

    """
def calculate_average(numbers):
    if len(numbers) == 0:
        return 0
    return sum(numbers) / len(numbers)
""",

    """
def count_items(items):
    return len(items)
""",

    """
def first_item(items):
    if len(items) == 0:
        return None
    return items[0]
""",

    """
def last_item(items):
    if len(items) == 0:
        return None
    return items[-1]
""",

    """
def reverse_list(items):
    return items[::-1]
""",

    """
def sort_numbers(numbers):
    return sorted(numbers)
""",

    """
def contains_item(items, value):
    return value in items
""",

    """
def count_value(items, value):
    return items.count(value)
""",

    """
def get_positive(numbers):
    return [x for x in numbers if x > 0]
""",

    """
def get_negative(numbers):
    return [x for x in numbers if x < 0]
""",

    """
def double_numbers(numbers):
    return [x * 2 for x in numbers]
""",

    """
def square_numbers(numbers):
    return [x * x for x in numbers]
""",

    """
def greet(name):
    return "Hello, " + name
""",

    """
def make_uppercase(text):
    return text.upper()
""",

    """
def make_lowercase(text):
    return text.lower()
""",

    """
def count_characters(text):
    return len(text)
""",

    """
def first_character(text):
    if len(text) == 0:
        return ""
    return text[0]
""",

    """
def last_character(text):
    if len(text) == 0:
        return ""
    return text[-1]
""",

    """
def reverse_text(text):
    return text[::-1]
""",

    """
def contains_text(text, word):
    return word in text
""",

    """
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
""",

    """
def count_to_n(n):
    numbers = []
    for i in range(1, n + 1):
        numbers.append(i)
    return numbers
""",

    """
def multiply_list(numbers, factor):
    result = []
    for number in numbers:
        result.append(number * factor)
    return result
""",

    """
def remove_negative(numbers):
    result = []
    for number in numbers:
        if number >= 0:
            result.append(number)
    return result
""",

    """
def find_even_numbers(numbers):
    result = []
    for number in numbers:
        if number % 2 == 0:
            result.append(number)
    return result
""",

    """
def find_odd_numbers(numbers):
    result = []
    for number in numbers:
        if number % 2 != 0:
            result.append(number)
    return result
""",

    """
def power(base, exponent):
    return base ** exponent
""",

    """
def percentage(value, total):
    if total == 0:
        return 0
    return value / total * 100
""",

    """
def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value
""",

    """
def is_positive(number):
    return number > 0
""",

    """
def is_negative(number):
    return number < 0
""",

    """
def is_zero(number):
    return number == 0
""",
]


# -----------------------------
# Combine examples
# -----------------------------

dataset = "\n".join(examples)

# Remove unnecessary blank space
dataset = dataset.strip()


# -----------------------------
# Split dataset
# -----------------------------

split_index = int(len(examples) * 0.8)

train_examples = examples[:split_index]
val_examples = examples[split_index:]

train_text = "\n".join(train_examples).strip()
val_text = "\n".join(val_examples).strip()


# -----------------------------
# Save files
# -----------------------------

base_path = Path(__file__).parent
data_path = base_path / "data"

data_path.mkdir(exist_ok=True)

with open(data_path / "code.txt", "w", encoding="utf-8") as file:
    file.write(dataset)

with open(data_path / "train.txt", "w", encoding="utf-8") as file:
    file.write(train_text)

with open(data_path / "val.txt", "w", encoding="utf-8") as file:
    file.write(val_text)


print("Total examples:", len(examples))
print("Training examples:", len(train_examples))
print("Validation examples:", len(val_examples))
print("Dataset split complete.")