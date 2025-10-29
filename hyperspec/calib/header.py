def read_header(path_file_header: str):
    assert path_file_header.endswith('.hdr')

    attrs = {}
    current_key = ''
    with open(path_file_header, 'r') as f:
        current_value = ''
        for line in f:
            line = line.strip('\n')
            if '=' not in line:
                current_value += line
            else:
                # submit previous entry
                # start new entry
                attrs[current_key.strip(' ')] = current_value.strip(' ')
                current_key, current_value = line.split('=', 1)
        # final submit
        attrs[current_key.strip(' ')] = current_value.strip(' ')
    return attrs

