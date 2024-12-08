from bs4 import BeautifulSoup

def parse_html_to_chart(file_path):
    """
    Parse the HTML file and convert it into a chart dictionary.
    
    Args:
        file_path (str): The path to the HTML file containing hand data.

    Returns:
        dict: A dictionary of hands with fold, call, and raise percentages.
    """
    # Read the HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    chart = {}

    for row in soup.find_all("div", class_="row"):
        hand = row["data-handid"]
        fold_percentage = 0.0
        raise_percentage = 0.0

        # Extract fold percentage
        fold_element = row.find("div", class_="row_fill color-fold")
        if fold_element:
            fold_percentage = float(fold_element["style"].split("width:")[1].split("%")[0]) / 100

        # Extract raise percentage
        raise_element = row.find("div", class_="row_fill color-raise")
        if raise_element:
            raise_percentage = float(raise_element["style"].split("width:")[1].split("%")[0]) / 100

        # Calculate call percentage (remaining percentage)
        call_percentage = 1.0 - fold_percentage - raise_percentage

        chart[hand] = {
            "fold": fold_percentage,
            "call": call_percentage,
            "raise": raise_percentage
        }

    return chart


# Example usage
file_path = "./chart.html"  # Replace with the path to your HTML file
parsed_chart = parse_html_to_chart(file_path)

# Print the parsed chart
print(parsed_chart)
