# Statistical Graph

1.茎叶图

```python
from itertools import groupby
data = '89 79 57 46 1 24 71 5 6 9 10 15 16 19 22 31 40 41 52 55 60 61 65 69 70 75 85 91 92 94'

for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 10):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print k, '|', ' '.join(lst)
```

0 | 1     
1 | 0 5 6 9                          
2 | 2 4     
3 | 1     
4 | 0 1 6    
0 | 5    
5 | 2 5 7   
0 | 6   
6 | 0 1 5 9   
7 | 0 1 5 9   
8 | 5 9   
0 | 9    
9 | 1 2 4   