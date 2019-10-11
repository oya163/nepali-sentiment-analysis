import numpy as np
data = json.load(open(file,'r',encoding='utf8'))

def check_devnagari(comment):
    count = re.findall("[\u0900-\u097F]+", comment)
    if (len(count)>0):
        return True
    return False

json_data = []
for item in data:
    comment = item['text']
    if (check_devnagari(comment)):
        json_data.append(item)

print("Total",len(json_data))
print(np.random.choice(json_data, 100, replace=False))
