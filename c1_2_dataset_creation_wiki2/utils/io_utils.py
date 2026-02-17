def read_text_from_file(file_path):
    """Read text content from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()