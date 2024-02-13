from datetime import datetime

# 一些字符串操作 24.2.13
## 1、在大数中添加3位分隔符
n: int = 1000000000
print(f'{n:_}')
print(f'{n:,}')
## 2、将字符串置于指定位置，如居中、左对齐或右对齐；并在空处补全其他字符
var: str = 'var'
print(f'{var:_>20}:')
print(f'{var:#<20}:')
print(f'{var:|^20}:')
## 3、格式化日期并按指定格式显示
now: datetime = datetime.now()
print(f'{now:%y.%m.%d，%H:%M:%S}')
print(f'{now:%I%p}')
## 4、保留小数至指定位数
n: float = 3.14159265358979323846
print(round(n, 2))
print(f'{n:.2f}')
print(f'{n * 1000:,.2f}')
## 5、同时输出计算式和计算结果
a: int = 5
b: int = 10
var: str = 'Hello World.'
print(f'{a + b = }')
print(f'{var = }')
