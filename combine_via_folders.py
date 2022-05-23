import json
import os
import shutil

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-paths', nargs='+',
                        help='folders with the images and via_region_data.json')
    parser.add_argument('-output',
                        help='output folder with the combined images and via_region_data.json')

    args = parser.parse_args()

    paths = args.paths
    output = args.output


    def load_json2dict(pth: str) -> dict:
        with open(pth) as json_file:
            return json.load(json_file)


    combined_dict = {}
    for pth in paths:
        via_region_data_pth = os.path.join(pth, "via_region_data.json")
        json_dict = load_json2dict(via_region_data_pth)
        combined_dict.update(json_dict)

    if os.path.exists(output):
        shutil.rmtree(output)

    if not os.path.exists(output):
        os.makedirs(output)

    for folder in paths:
        for file in os.listdir(folder):
            if file.endswith(".jpeg") or file.endswith(".jpg"):
                shutil.copy(os.path.join(folder, file), os.path.join(output, file))

    combined_json_file = open(os.path.join(output, "via_region_data.json"), "w")
    json.dump(combined_dict, combined_json_file)

    from zipfile import ZipFile

    dirname = os.path.dirname(output)
    with ZipFile(f'{os.path.join(output, dirname)}.zip', 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(dirname):
            for filename in filenames:
                # create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath, os.path.basename(filePath))

