
def reverse_text(text):
    """
    Reverses the given text.
    Args:
        text (str): Input text.
    Returns:
        str: Reversed text.
    Raises:
        ValueError: If input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    return text[::-1]


def capitalize_text(text):
    """
    Capitalizes the given text.
    Args:
        text (str): Input text.
    Returns:
        str: Capitalized text.
    Raises:
        ValueError: If input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    return text.capitalize()


def count_words(text):
    """
    Counts the number of words in the text.
    Args:
        text (str): Input text.
    Returns:
        int: Number of words.
    Raises:
        ValueError: If input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    return len(text.split())


def remove_duplicates(text):
    """
    Removes duplicate words while preserving order.
    Args:
        text (str): Input text.
    Returns:
        str: Text with duplicate words removed.
    Raises:
        ValueError: If input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    seen = set()
    result = []

    for word in text.split():
        if word not in seen:
            seen.add(word)
            result.append(word)

    return " ".join(result)


def to_uppercase(text):
    """
    Converts the given text to uppercase.
    Args:
        text (str): Input text.
    Returns:
        str: Uppercase text.
    Raises:
        ValueError: If input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    return text.upper()


def to_lowercase(text):
    """
    Converts the given text to lowercase.
    Args:
        text (str): Input text.
    Returns:
        str: Lowercase text.
    Raises:
        ValueError: If input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    return text.lower()


def count_characters(text):
    """
    Counts the number of characters in the text excluding spaces.
    Args:
        text (str): Input text.
    Returns:
        int: Number of characters excluding spaces.
    Raises:
        ValueError: If input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    return len(text.replace(" ", ""))


def remove_spaces(text):
    """
    Removes all spaces from the text.
    Args:
        text (str): Input text.
    Returns:
        str: Text without spaces.
    Raises:
        ValueError: If input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    return text.replace(" ", "")


def is_palindrome(text):
    """
    Checks if the given text is a palindrome.
    Args:
        text (str): Input text.
    Returns:
        bool: True if palindrome, False otherwise.
    Raises:
        ValueError: If input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    cleaned_text = text.replace(" ", "").lower()
    return cleaned_text == cleaned_text[::-1]


def title_case(text):
    """
    Converts the given text to title case.
    Args:
        text (str): Input text.
    Returns:
        str: Title-cased text.
    Raises:
        ValueError: If input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    return text.title()


def starts_with(text, prefix):
    """
    Checks if the text starts with the given prefix.
    Args:
        text (str): Input text.
        prefix (str): Prefix to check.
    Returns:
        bool: True if text starts with prefix, False otherwise.
    Raises:
        ValueError: If inputs are not strings.
    """
    if not isinstance(text, str) or not isinstance(prefix, str):
        raise ValueError("Both inputs must be strings.")
    return text.startswith(prefix)


def ends_with(text, suffix):
    """
    Checks if the text ends with the given suffix.
    Args:
        text (str): Input text.
        suffix (str): Suffix to check.
    Returns:
        bool: True if text ends with suffix, False otherwise.
    Raises:
        ValueError: If inputs are not strings.
    """
    if not isinstance(text, str) or not isinstance(suffix, str):
        raise ValueError("Both inputs must be strings.")
    return text.endswith(suffix)


# text = "hello world hola"
# rev = reverse_text(text)
# cap = capitalize_text(text)
# wc = count_words(text)
# rd = remove_duplicates(text)
# up = to_uppercase(text)
# low = to_lowercase(text)
# cc = count_characters(text)
# rs = remove_spaces(text)
# pal = is_palindrome("madam")
# title = title_case(text)
# sw = starts_with(text, "hello")
# ew = ends_with(text, "hello")
