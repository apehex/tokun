# METADATA ####################################################################

def label(token: str) -> str:
    return '#{}'.format(token.encode('utf-32-be').hex())

# SERIALIZE ###################################################################

def write(data: any, path: str, tsv: bool=True) -> None:
    with open(path, 'w') as __f:
        for __row in data:
            __line = '\t'.join(str(__v) for __v in __row) if tsv else str(__row)
            __f.write(__line + '\n')
