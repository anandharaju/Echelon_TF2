# Code to parse PE file's available sections using 'pefile' 3rd party library
# Input: Name of the file to be parsed
# Output: Byte representation of section level data in PE file

import pefile
import logging
from config import constants as cnst

'''def parse_pe(file):
    pe = pefile.PE(file)
    for section in pe.sections:
        #if not section.Name.find(b'.text'):
            print("***************", section.Name.rstrip(b'\x00'), hex(section.VirtualAddress), hex(section.Misc_VirtualSize), section.SizeOfRawData, section.PointerToRawData)
            print(len(section.get_data()), section.get_data())'''

# logging.basicConfig(filename='..'+cnst.ESC+'log'+cnst.ESC+'parse_pe'+cnst.ESC+'parse_pe.log', filemode='w+')


def parse_pe(i, file, max_len, unprocessed):
    try:
        pe = pefile.PE(file)
        sections = [(".header", 0, len(pe.header))]
        last_section_ends_at = None
        for item in pe.sections:
            #print(section.PointerToRawData, section.SizeOfRawData)
            sections.append((item.Name.rstrip(b'\x00').decode("utf-8").strip(),
                         item.PointerToRawData, item.PointerToRawData + item.SizeOfRawData - 1))
            last_section_ends_at = item.PointerToRawData + item.SizeOfRawData - 1
        sections += [(".padding", last_section_ends_at + 1, 512000)]
    except Exception as e:
        msg = str(e) + " [FILE ID - " + str(i) + "]  [" + file + "] "
        logging.error(msg)
        unprocessed += 1
    return sections, unprocessed


def parse_pe_section_data(file, section_to_extract):
    try:
        extracted_details = bytes([0])
        pe = pefile.PE(file)
        if section_to_extract == '.header':
            return pe.header
        #print(pe.dump_info())
        for section in pe.sections:
            #print(section.Name.rstrip(b'\x00'), hex(section.VirtualAddress), hex(section.Misc_VirtualSize),section.SizeOfRawData, section.PointerToRawData)
            section_name = str(section.Name).strip()
            if section_name.find(section_to_extract) > 0:
                extracted_details = section.get_data()
                #print(section.Name.rstrip(b'\x00'), hex(section.VirtualAddress), hex(section.Misc_VirtualSize), section.SizeOfRawData, section.PointerToRawData)
                #print(len(section.get_data()), len(section.get_data().rstrip(b'\x00')), len(section.get_data()) - len(section.get_data().rstrip(b'\x00')))
    except Exception as e:
        msg = str(e) + " [FILE ID - " + file + "] "
        logging.error(msg)
    return extracted_details


# parse_pe("MsiTrueColorHelper.exe")
# pe = pefile.PE("git-gui.exe")
# print(pe.header)
# print(pe.OPTIONAL_HEADER)
# print(pe.NT_HEADERS)
# print(pe.FILE_HEADER)
# print(pe.RICH_HEADER)
# print(pe.DOS_HEADER)
# for section in pe.sections:
#    print(section)

