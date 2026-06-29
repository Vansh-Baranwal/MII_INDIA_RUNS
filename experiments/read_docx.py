import zipfile
import xml.etree.ElementTree as ET
import sys
import os

def read_docx(filename):
    try:
        doc = zipfile.ZipFile(filename)
        xml_content = doc.read('word/document.xml')
        doc.close()
        tree = ET.XML(xml_content)
        
        NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        PARA = NAMESPACE + 'p'
        TEXT = NAMESPACE + 't'
        
        paragraphs = []
        for paragraph in tree.iter(PARA):
            texts = [node.text
                     for node in paragraph.iter(TEXT)
                     if node.text]
            if texts:
                paragraphs.append(''.join(texts))
        return '\n'.join(paragraphs)
    except Exception as e:
        return f"Error reading {filename}: {e}"

base_dir = r"d:/Some_stuffs/India Runs/[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge"
files = ['README.docx', 'job_description.docx', 'redrob_signals_doc.docx', 'submission_spec.docx']
out_path = os.path.join(base_dir, 'read_docx_output.txt')

with open(out_path, 'w', encoding='utf-8') as f:
    for fname in files:
        filepath = os.path.join(base_dir, fname)
        if os.path.exists(filepath):
            f.write("="*50 + "\n")
            f.write("FILE: " + fname + "\n")
            f.write("="*50 + "\n")
            f.write(read_docx(filepath) + "\n\n")
