import re
LITERAL_REGEX = re.compile(r'"([^"\\\n]*(?:\\.[^"\\\n]*)*)"')
print(LITERAL_REGEX.findall('for_of ::= "for" " " "(" "const"'))
