import sys

def replace_newlines(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return f"Error: File '{filename}' not found."
    except Exception as e:
        return f"Error: An error occurred while reading the file: {e}"

    modified_content = content.replace('\n\n', 'TEMP_DOUBLE_NEWLINE').replace('\n', ' ').replace('TEMP_DOUBLE_NEWLINE', '\n\n')

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(modified_content)
    except Exception as e:
        return f"Error: An error occurred while writing to the file: {e}"

    return "Newlines replaced successfully."

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python replace_newlines.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    result = replace_newlines(filename)
    print(result)

    # Test case
    test_filename = "test_file.txt"
    with open(test_filename, 'w', encoding='utf-8') as f:
        f.write("This is a test.\nThis is a single newline.\n\nThis is a double newline.\nThis is another single newline.\n\n")

    replace_newlines(test_filename)

    with open(test_filename, 'r', encoding='utf-8') as f:
        test_content = f.read()

    print(f"Test result: {test_content}")
