C_struct = "./toto.txt"

with open(C_struct) as f:
    data = f.read()
data = data.split('\n')
nn=len(data)
# t = [row.split('\t')[0] for row in data[1:]]
s = "__fields__=["
for row in data[:]:
    _row = row.split(" ")
    _row  = filter(None, _row ) # fastest
    if _row and _row[0][0:2] != "//":
        print _row[0][0:1]
        _row[1] = _row[1].replace(";", "")
        _row[0] = _row[0].replace("SHORT", "ctypes.c_short")
        _row[0] = _row[0].replace("REAL", "ctypes.c_double")
        _row[0] = _row[0].replace("INT", "ctypes.c_int")
        if _row[1][0] == '*':
            _row[1] = _row[1].replace("*", "")
            _row[0] = _row[0].replace("ctypes.c_double", "ctypes.POINTER(ctypes.c_double)")
            _row[0] = _row[0].replace("ctypes.c_int", "ctypes.POINTER(ctypes.c_int)")
        s += '(\"'+_row[1]+'\" , '+_row[0]+'), \n'

s +=  ']'
