import xml.etree.ElementTree as ET
import sys

def parse_xml(file_path):
    try:
        tree = ET.parse(file_path)
        return tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None


#Recursively  func to extract text within text type shit
def extract_text(element):
    text = element.text or ""
    for child in element:
        text += extract_text(child)
        text += child.tail or ""
    return text

#Actual parsing
def extract_information(xml_root):
    # creating an dictionary that is just a repository of info extracted
    extracted_info = {}

    # Extract basic info
    extracted_info['BillNumber'] = xml_root.find(".//BillNumber").text
    extracted_info['Session'] = xml_root.find(".//Session").text
    extracted_info['ParliamentNumber'] = xml_root.find(".//Number").text

    #Using long title if no short title
    short_title_element = xml_root.find(".//ShortTitle[@status='unofficial']")
    long_title_element = xml_root.find(".//LongTitle")
    extracted_info['Title'] = extract_text(long_title_element) if long_title_element is not None else extract_text(short_title_element)


    #Getting DMY
    extracted_info['AssentedYear'] = xml_root.find(".//Stages[@stage='assented-to']/Date/YYYY").text
    extracted_info['AssentedMonth'] = xml_root.find(".//Stages[@stage='assented-to']/Date/MM").text
    extracted_info['AssentedDay'] = xml_root.find(".//Stages[@stage='assented-to']/Date/DD").text


    # Requires a whole different process due to all the element types, but basically were scannign for whichever ones already have text and getting text from them
    extracted_info['Body'] = {}
    current_heading = None
    for item in xml_root.findall(".//Body/*"):
                # for every heading
        if item.tag == 'Heading':
            current_heading = item.find('TitleText').text
        elif item.tag == 'Section':
            # Extracting the text from children
            section_content = []

            # Extract MarginalNote text
            marginal_note = item.find('.//MarginalNote')
            if marginal_note is not None:
                section_content.append(f"{extract_text(marginal_note)}")

            # Extracting the text from children
            section_texts = [extract_text(text) for text in item.findall(".//Text")]
            section_content.extend(section_texts)

            # Creating a new heading if one shows up and adding to dictionary
            if current_heading not in extracted_info['Body']:
                extracted_info['Body'][current_heading] = []
                
             # Adding section and marginal notes to its respective heading
            extracted_info['Body'][current_heading].extend(section_content)

    return extracted_info



# main part of code starts here, I should progably acc have a main function

xml_file_path = "S-5_E.xml"

# Takign the file path and getting root, if the root is there than pasing
xml_root = parse_xml(xml_file_path)

if xml_root:
    extracted_data = extract_information(xml_root)

output_file_path = "output_with_marginal_notes.txt"

# using printing to output file in the same way
#utf 8 encoding for special chars
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    sys.stdout = output_file

    # throwing hte shit in the file
     # iterates through items like date and body 
    for key, value in extracted_data.items():
        if key == 'Body':
            print(f"{key}:")
            # body is slightliy more annoying bc its now a 3d structure, iterates through its items
            # now headigns are the keys, sections are the values
            for heading, sections in value.items():
                print(f"  {heading}:")
                for section in sections:
                    print(f"    {section}")
        else:
            print(f"{key}: {value}")

sys.stdout = sys.__stdout__

print(f"Output written to {output_file_path}")
