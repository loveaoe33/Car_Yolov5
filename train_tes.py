import xml.etree.ElementTree as ET

def generate_tesseract_training_data(xml_file, output_box_file, output_text_file):
    tree=ET.parse(xml_file)
    root=tree.getroot()

    with open(output_box_file, 'w') as box_file, open(output_text_file, 'w') as text_file:
        for obj in root.findall("object"):
            label = obj.find('name').text
            bbox = obj.find('bndbox')

            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            box_file.write(f'{xmin} {ymin} {xmax} {ymax} 0\n')
            text_file.write(f'{label}\n')


xml_file="C:/Users/loveaoe33/Desktop/text_train"
output_box_file="C:\\Users\\loveaoe33\\Desktop\\output_box_file"
output_text_file="C:\\Users\\loveaoe33\\Desktop\\output_text_file"
generate_tesseract_training_data(xml_file, output_box_file, output_text_file)
