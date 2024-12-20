from helpers.vlm import get_image_description
import json, os, re, io, base64
from helpers.utils import get_hash
from io import StringIO
from PIL import Image
import pymupdf4llm
import requests
import pathlib




class PdfReader():

    cache: dict

    def __init__(self):
        self.cache = {}
    def __has_image(self, file_content: str) -> bool:
        return bool(re.search(r"!\[\]\(*.*\)", file_content))

    def __embed_image_description(self, input_file_content: str) -> str:
        tempfile = StringIO()
        encoded_images = []
        for line in input_file_content.split('\n'):
            if re.match(r"!\[\]\(*.*\)", line):
                # get the file name
                filepath = re.search(r"\([^)]*\)", line).group(0)[1:-1]
                # check if file exists or not
                if not os.path.isfile(filepath):
                    print("[-] File {} does not exists".format(filepath))
                    # writing the original line to the file 
                    tempfile.write(line)
                    continue
                FileExtension = pathlib.Path(filepath).suffix.lower()
                # filtering the extensions
                extens = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']
                Matched = False
                for ext in extens:
                    if FileExtension == ext:
                        Matched = True
                        line2write = f'<imgdesc>{get_image_description(filepath)}</imgdesc>'
                        tempfile.write(line2write+'\n')
                        encoded_images.append(filepath)
                        break
                if not Matched:
                    print("[-] Extension is not allowd: {}".format(FileExtension))
                    print("[-] Filepath {} is not written into Output file.".format(filepath))
            else:
                # write the data into a new file 
                tempfile.write(line)
        # close the file
        contents = tempfile.getvalue()
        tempfile.close()
        return contents

    def __get_pdf_content_with_imgdesc(self, file_path: str) -> tuple[str, bool]:
        file_name = os.path.basename(file_path)
        file_name_without_ext = os.path.splitext(file_name)[0]
        md_text = pymupdf4llm.to_markdown(file_path, write_images=True, table_strategy='lines', image_path=file_name_without_ext)
        if self.__has_image(md_text):
            return self.__embed_image_description(md_text), True
        else:
            return md_text, False
    
    def __get_content_from_cache(self, file_path) -> str:
        file_hash = get_hash(file_path)
        if file_hash in self.cache:
            return self.cache[file_hash]
        content, _ = self.__get_pdf_content_with_imgdesc(file_path)
        self.cache[file_hash] = content
        return content
        
    def get_pdf_content(self, file_path: str) -> str:
        """Reads the content of a PDF file and returns the content as a string, associated image's description are provided inside <imgdesc></imgdesc> tags. Takes the file_path as argument"""
        print(file_path)
        file_path = file_path.strip()
        file_path = os.path.normpath(file_path)
        content = self.__get_content_from_cache(file_path)
        return content

    def get_pdf_content_from_web(self, url: str) -> str:
        """Reads the content of a PDF file from a web url and returns the content as a string, associated image's description are provided inside <imgdesc></imgdesc> tags. Takes the url as argument"""
        print(url)
        url = url.strip()
        if url in self.cache:
            return self.cache[url]
        else:
            response = requests.get(url)
            if response.headers.get('Content-Type') != 'application/pdf':
                return "Can not fetch file from this url"
            if response.headers.get('Content-Length') is None:
                return "Can not fetch file from this url"
            if response.status_code != 200:
                return "Can not fetch file from this url"
            if int(response.headers.get('Content-Length')) > 10000000:
                return "File size is too large"
            if int(response.headers.get('Content-Length')) == 0:
                return "File is empty"
            # save the file to a temporary file
            file_path = os.path.join(os.getcwd(), url.split('/')[-1])
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return self.get_pdf_content(file_path)
