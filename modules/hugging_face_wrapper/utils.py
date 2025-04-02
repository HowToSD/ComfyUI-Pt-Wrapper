import re
from html import unescape


def drop_html_tags(inputs:str) -> str:
    """
    Removes html tags from input.

    Args:
        inputs (str): Input text.

    Returns:
        str: Text without html tags.
    """
    
    # Decode HTML entities like "&amp;", "&lt;", etc. to their actual characters
    text = unescape(inputs)
    
    # Remove HTML tags (like <br />, <p>, etc.)
    # The regex "<.*?>" uses .*? (non-greedy match) to ensure it matches one tag at a time,
    # instead of greedily removing everything between the first '<' and last '>'
    text = re.sub(r"<.*?>", " ", text)

    # Replace consecutive white space to a single white space
    text = re.sub(r"  +", " ", text)

    return text