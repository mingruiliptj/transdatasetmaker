import sys
import re

def replace_numeric_lines(filename):
    """
    Replaces lines containing only numbers with whitespace in a given file.
    Writes the modified content to a new file with '_modified' appended to the original filename.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        modified_lines = []
        for line in lines:
            # Check if the line contains only digits after removing leading/trailing whitespace
            if re.match(r'^\s*\d+\s*$', line):
                modified_lines.append('\n')  # Replace with a newline (whitespace)
            else:
                modified_lines.append(line)

        # Create a new filename for the modified file
        output_filename = filename.rsplit('.', 1)[0] + '_modified.' + filename.rsplit('.', 1)[1]

        with open(output_filename, 'w', encoding='utf-8') as outfile:
            outfile.writelines(modified_lines)

        print(f"Modified file saved as: {output_filename}")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python replace_numeric_lines.py <filename>")
    else:
        filename = sys.argv[1]
        replace_numeric_lines(filename)
