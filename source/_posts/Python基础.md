---
title: Python基础
date: 2019-02-05 18:51:15
tags:
---
### 列表 
list是一种有序的集合，可以随时添加和删除其中的元素
```
list =[] 
# 空列表

list.append('Google')
# 添加元素，对列表自己内部进行操作， 不会有返回值

list.extend(seq)
# 列表末尾一次性追加另一个序列中的多个值,原列表内部操作，不会有返回值

del list[2]
# 删除元素，对列表自己内部进行操作， 不会有返回值

del list[:]
# 删除所有元素

list.count(obj)
# 统计某个元素在列表中出现的次数

list.index(obj)
# 从列表中找出某个值第一个匹配项的索引位置

list.insert(index, obj)
# 将对象插入列表,对列表自己内部进行操作， 不会有返回值

list.pop([index=-1])
# 移除列表中的一个元素（默认最后一个元素），并且返回该元素值

list.remove(obj)
#移除列表中某个值的第一个匹配项,对列表自己内部进行操作，不会有返回值

list.reverse()
# 反向列表中元素,不会有返回值

list.sort(cmp=None, key=None, reverse=False)
# 对原列表进行排序,不会有返回值

sorted(list,key=,reverse=)
# 不改变原列表，返回新列表

```



### IO 模块
open()函数用于打开一个文件，创建一个file对象  ,通过file对象，可以得到有关该文件的各种信息

```
file=open(filename,'r+')
# 打开文件用于读写

file=open(filename,'w+')
# 打开文件用于读写，若文件存在则覆盖，若不存在，则创建文件

file=open(filename,'a+')
# 打开文件用于读写，若文件存在，文件末尾追加，若不存在，则创建文件

file_read=file.read([count])
# 从文件的开头开始读入,count为字节数

file_read=file.readline([size])
# 返回包含size行的列表,size 未指定则返回全部行

file_write=file.write(string)
# write()方法不会在字符串的结尾添加换行符('\n')

file=file.close()
# 关闭文件

for line in file:
# file对象可迭代

with open(filename) as f:
# with语句可自动调用file的close()方法


```


### OS模块
Python 的 os 模块封装了常见的文件和目录操作

> os: This module provides a portable way of using operating system dependent functionality.
> 这个模块提供了一种方便的使用操作系统函数的方法

> sys: This module provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
> 这个模块可供访问由解释器使用或维护的变量和与解释器进行交互的函数。



| 方法 | 说明 |
| ------ | ------ |
| os.mkdir  |	创建目录 |
|os.rmdir|	 删除目录 |
|os.chdir(path)| 更改目录|
|os.rename|	 重命名|
|os.remove|	删除文件|
|os.getcwd|	获取当前工作路径|
|os.walk|	遍历目录|
|os.path.join|	连接目录与文件名|
|os.path.split|	分割文件名与目录|
|os.path.abspath|	获取绝对路径|
|os.path.dirname|	获取路径|
|os.path.basename|	获取文件名或文件夹名|
|os.path.splitext|	分离文件名与扩展名|
|os.path.isfile	|判断给出的路径是否是一个文件|
|os.path.isdir	|判断给出的路径是否是一个目录|



|方法	|说明|
| :------: | :------: |
|sys.argv| 命令行参数List，第一个元素是程序本身路径|
|sys.modules.keys()| 返回所有已经导入的模块列表|
|sys.exc_info()| 获取当前正在处理的异常类,exc_type、exc_value、exc_traceback当前处理的异常详细信息|
|sys.exit(n)| 退出程序，正常退出时exit(0)|
|sys.hexversion| 获取Python解释程序的版本值，16进制格式如：0x020403F0|
|sys.version| 获取Python解释程序的版本信息|
|sys.maxint| 最大的Int值|
|sys.maxunicode| 最大的Unicode值|
|sys.modules| 返回系统导入的模块字段，key是模块名，value是模块|
|sys.path |返回模块的搜索路径，初始化时使用PYTHONPATH环境变量的值|
|sys.platform |返回操作系统平台名称|
|sys.stdout |标准输出|
|sys.stdin |标准输入|
|sys.stderr |错误输出|
|sys.exc_clear() |用来清除当前线程所出现的当前的或最近的错误信息|
|sys.exec_prefix |返回平台独立的python文件安装的位置|
|sys.byteorder |本地字节规则的指示器，big-endian平台的值是'big',little-endian平台的值是'little'|
|sys.copyright |记录python版权相关的东西|
|sys.api_version| 解释器的C的API版本|

### re模块​    
#### re.match(pattern, string [, flags] )     

 需完全匹配出表达式，才成功，从string头开始到pattern匹配结束，同时匹配终止，不再匹配后面字符

> import re  
> pattern= re.compile(r'hello')  
> result1=re.match(pattern,'hello')  
> print result1.group()

 match对象的属性

- sring：匹配时使用的文本
- re：匹配时使用的正则对象
- group() :返回分组截获的字符串
- groups()：以元组形式返回全部分组截获的字符串
- groupdict():返回有别名的组，字典形式展现
- expand(template) 将匹配到的分组代入template中然后返回，template可以用 \id或\g 引用分组

#### re.search(pattern, string [, flags])

 search方法与match方法区别：在于match() 函数从string开始位置匹配，search() 会扫描整个string查找匹配

#### re.split(pattern, string [,maxsplit])

按照能够匹配的子串将string分割后返回列表。maxsplit用于指定最大分割次数，不指定将全部分割   

> import re  
> pattern= re.compile(r'\d+')  
> print re.split(pattern,'one1two2three3four4')  
> ['one', 'two', 'three', 'four', '']    

#### re.findall(pattern, string[, flags]) 

搜索string，以列表形式返回全部能匹配的子串

> import re  
> pattern = re.compile(r'\d+')  
> print re.findall(pattern,'one1two2three3four4')  
> ['1', '2', '3', '4']

#### re.finditer(pattern, string[, flags])

搜索string，返回一个顺序访问每一个匹配结果（Match对象）的迭代器

> import re  
> pattern = re.compile(r'\d+')  
> for m in re.finditer(pattern,'one1two2three3four4'):  
> print m.group()

#### re.sub(pattern, repl, string[, count])
使用repl替换string中每一个匹配的子串后返回替换后的字符串

#### re.subn(pattern, repl, string[, count])

返回 (sub(repl, string[, count]), 替换次数)

  


### 函数

- execfile\(\)函数：用来执行一个文件

- file\(\)函数：用来创建一个file对象

- filter(function, iterable) 函数:

  ```
  >>>daily_spend = [110.32, 0, 445.32, 0, 88.83,0]
  >>>has_spend = filter(lambda x: x!=0,daily_spend)
  >>>print(list(has_spend))
  [110.32, 445.32, 88.83]
  ```

- iter\(\)函数：用来生成迭代器

- map(function, iterable, ...)函数会根据提供的函数对指定序列做映射

```
>>> map(lambda x: x+1,[1,2,3,4]) 
[2, 3, 4, 5]  
>>> map(lambda x,y: x+y,[1,2,3,4],(10,20,30,40))
[11, 22, 33, 44]
```

- reduce(function, iterable[, initializer]) 函数会对参数序列中元素进行累积

```
>>>daily_spend = [110.32, 0, 445.32, 0, 88.83,0]
>>>print(reduce(lambda x,y:x+y, daily_spend))
644.47
```

- zip函数：将多个序列相同位置上的元素，组合成元组后放入列表中

```
>>>days, spend = [10,25,30], [12.20, 20.43, 44.32]
>>>consumptions = zip(days, spend)
>>>print(list(consumptions ))
[(10, 12.2), (25, 20.43), (30, 44.32)]
```

- sort\(\)函数应用在list上，对列表自己内部进行排序， 不会有返回值， 因此返回为None，sorted\(\)函数应用在所有可迭代对象，产生新的对象
- set\(\) 函数创建一个无序不重复元素集，可进行关系测试
- slice\(\) 函数实现切片对象
- strip\(\) 方法用于移除字符串头尾指定的字符（默认删除所有空白符）
- Python内置的enumerate函数可以把一个list变成索引-元素对

- 生成器：generator，是种一边循环一边计算的机制，generator保存的是算法，可以通过next()函数获得generator的下一个返回值
- 迭代器：Iterator表示的是一个数据流，对象可以被next()函数调用并不断返回下一个数据，可以把这个数据流看做是一个有序序列，但我们却不能提前知道序列的长度，只能不断通过next()函数实现按需计算下一个数据
- yield函数，在调用生成器运行的过程中，每次遇到 yield 时函数会暂停并保存当前所有的运行信息，返回 yield 的值, 并在下一次执行 next() 方法时从当前位置继续运行。

- format 格式化函数
- 'sep'.join\(\) 返回一个以分隔符sep连接各个元素后生成的字符串

```
"{} {}".format("hello", "world")    # 不设置指定位置，按默认顺序
'hello world'

 "{0} {1}".format("hello", "world")  # 设置指定位置
'hello world'

 "{1} {0} {1}".format("hello", "world")  # 设置指定位置
'world hello world'

 site = {"name": "菜鸟教程", "url": "www.runoob.com"}
 print("网站名：{name}, 地址 {url}".format(**site))

 my_list = ['菜鸟教程', 'www.runoob.com']
 print("网站名：{0[0]}, 地址 {0[1]}".format(my_list))  # "0" 是可选的
```

### Notion事项

- 多行注释使用三个单引号(''')或三个双引号(""")    
- 元组用"()"标识。内部元素用逗号隔开。但是元组不能二次赋值，相当于只读列表,元组中的元素值是不允许删除的，但我们可以使用del语句来删除整个元组     
- continue 用于跳过该次循环，break 则是用于退出循环    
- pass是空语句，是为了保持程序结构的完整性,pass 不做任何事情，一般用做占位语句   
- Python不支持单字符类型，单字符也在Python也是作为一个字符串使用    
- r/R	原始字符串,所有的字符串都是直接按照字面的意思来使用，没有转义特殊或不能打印的字符。 原始字符串除在字符串的第一个引号前加上字母"r"（可以大小写）以外，与普通字符串有着几乎完全相同的语法    
- 在 python 中，strings, tuples, 和 numbers 是不可更改的对象，而 list,dict 等则是可以修改的对象 
- set和dict的唯一区别仅在于没有存储对应的value
- seek（offset [,from]）方法改变当前文件的位置。Offset变量表示要移动的字节数。From变量指定开始移动字节的参考位置。
  如果from被设为0，这意味着将文件的开头作为移动字节的参考位置。如果设为1，则使用当前的位置作为参考位置。如果它被设为2，那么该文件的末尾将作为参考位置
- input() 和 raw_input() 这两个函数均能接收 字符串 ，但 raw_input() 直接读取控制台的输入（任何类型的输入它都可以接收）。而对于 input() ，它希望能够读取一个合法的 python 表达式，即你输入字符串的时候必须使用引号将它括起来
- tuple所谓的“不变”是说，tuple的每个元素，指向永远不变。list的元素可变
- 函数返回值，可以用return语句返回，函数体内部的语句在执行时，一旦执行到return时，函数就执行完毕，并将结果返回，函数执行完毕也没有return语句时，自动return None
- Python的函数返回多值其实就是返回一个tuple

- 不可变类型参数传递：整数、字符串、元组。

  如fun（a），传递的只是a的值，没有影响a对象本身。比如在 fun（a）内部修改 a 的值，只是修改另一个复制的对象，不会影响 a 本身。

- 可变类型参数传递：如列表，字典。

  如 fun（la），则是将 la 真正的传过去，修改后fun外部的la也会受影响

- 定义可变参数。可变参数就是传入的参数个数是可变的，仅仅在参数前面加了一个*号

  可变参数允许你传入0个或任意个参数，这些可变参数在函数调用时自动组装为一个tuple。而关键字参数允许你传入0个或任意个含参数名的参数，这些关键字参数在函数内部自动组装为一个dict

- *args是可变参数，args接收的是一个tuple

- **kw是关键字参数，kw接收的是一个dict

- 可变参数既可以直接传入：func(1, 2, 3)，又可以先组装list或tuple，再通过*args传入：func(*(1, 2, 3))

- 关键字参数既可以直接传入：func(a=1, b=2)，又可以先组装dict，再通过* * kw传入：func(**{'a': 1, 'b': 2})

- 实例的变量名如果以__开头（两个下划线），就变成了一个私有变量（private），只有内部可以访问，外部不能访问

- `__slots__`变量，可限制该class实例能添加的属性

- `__init__`方法的第一个参数永远是self，表示创建的实例本身，因此，在`__init__`方法内部，就可以把各种属性绑定到self，因为self就指向创建的实例本身

- imp.load_source('module_name','/path/file.py')
  将file.py 文件导入到 module_name中，module_name 可以自定义

- 每个Python脚本在运行时都有一个“name”属性。如果脚本作为模块被导入，则其“name”属性的值被自动设置为模块名；如果脚本独立运行，则其“name”属性值被自动设置为“main”

- python查找变量是顺序是：先局部变量，再全局变量。

- 直接赋值：其实就是对象的引用（别名），即变量指向同一对象

- 浅拷贝(copy)：拷贝父对象，不会拷贝对象的内部的子对象，即内部子对象做同步变化

- 深拷贝(deepcopy)： copy 模块的 deepcopy 方法，完全拷贝了父对象及其子对象。

- super() 函数是用于调用父类(超类)的一个方法

- isinstance(a, list) 判断变量类型


### itertools模块

```
 count(start=0, step=1)
# 创建一个迭代器，生成从n开始的连续整数

cycle(iterable)
# 创建一个能在一组值间无限循环的迭代器

repeat(object[, times])
# 创建一个迭代器，重复生成object，times（如果已提供）指定重复计数

chain(*iterables)
# 将多个迭代器作为参数, 但只返回单个迭代器, 它产生所有参数迭代器的内容, 就好像他们是来自于一个单一的序列

compress(data, selectors)
# 提供一个选择列表，对原始数据进行筛选

dropwhile(predicate, iterable)
# 创建一个迭代器，只要函数predicate(item)返回False，就会生成iterable中的项和所有后续项

takewhile(predicate, iterable)
# 创建一个迭代器，生成iterable中predicate(item)为True的项，只要predicate计算为False，迭代就会立即停止

groupby(iterable[, key])
# 返回一个产生按照key进行分组后的值集合的迭代器,生成元素(key, group)，其中key是分组的键值，group是迭代器，生成组成该组的所有项

ifilter(predicate, iterable)
# 返回当测试函数返回true时的项

islice(iterable, start, stop[, step])
# 如果省略了start，迭代将从0开始，如果省略了step，步幅将采用1.

imap(function, *iterables)
# 返回序列每个元素被func执行后返回值的序列的迭代器

starmap(function, iterable)
# 创建一个迭代器，生成值func(*item),其中item来自iterable

tee(iterable[, n=2])
# 从一个可迭代对象创建 n 个迭代器

izip(*iterables)
# 类似于内置函数zip(), 只是它返回的是一个迭代器而不是一个列表

product(*iterables[, repeat])
# 创建一个迭代器，生成表示item1，item2等中的项目的笛卡尔积的元组，repeat是一个关键字参数，指定重复生成序列的次数

permutations(iterable[, r])
# 返回iterable中所有长度为r的项目序列，如果省略了r，那么序列的长度与iterable中的项目数量相同

combinations(iterable, r)
# 创建一个迭代器，返回iterable中所有长度为r的子序列，返回的子序列中的项按输入iterable中的顺序排序

combinations_with_replacement(iterable, r)
# 与 combinations 非常类似。唯一的区别是，它会创建元素自己与自己的组合

```

### 字典

字典的键一般是唯一的，如果重复最后的一个键值对会替换前面的，值不需要唯一；键必须不可变，所以可以用数字，字符串或元组充当，所以用列表就不行

```
del dict['Name']
## 删除键是'Name'的条目

dict.clear()  
## 清空词典所有条目

del dict 
## 删除词典

str(dict)
## 返回string

dict.copy()
## 返回一个字典的浅复制

dict.fromkeys(seq[, val])
## 创建一个新字典，以序列 seq 中元素做字典的键，value 为字典所有键对应的初始值

dict.get(key, default=None)
## 返回指定键的值，如果值不在字典中返回默认值

dict.has_key(key)
## 如果键在字典dict里返回true，否则返回false

dict.items()
## 以列表返回可遍历的(键, 值) 元组数组

dict.keys()
## 以列表返回一个字典所有的键

dict.values()
## 以列表返回字典中的所有值

dict.setdefault(key, default=None)
## 如果字典中包含有给定键，则返回该键对应的值，否则返回为该键设置的值

dict.update(dict2)
## 把字典dict2的键/值对更新到dict里
	
pop(key[,default])
## 删除字典给定键 key 所对应的值，返回值为被删除的值。key值必须给出。 否则，返回default值。

popitem()
## 随机返回并删除字典中的一对键和值。

```

### 正则

```
*
# 零次或多次匹配前面的字符或子表达式

+
# 一次或多次匹配前面的字符或子表达式

？
# 零次或一次匹配前面的字符或子表达式

.
# 匹配除\r\n 之外的任何单个字符

{n}
# n是非负整数，正好匹配n次

{n,}
# 至少匹配n次

\d 
# 数字[0-9]

\D
# 非数字

\s
# 空白字符

\S
# 非空白字符

\w
# 单词字符[A-Za-z0-9_]（字母、数字、下划线、汉字）

\W
# 非单词字符



(?P<name>正则表达式) 
#命名分组，name是一个合法的标识符

group(num)和groups()
#获得分组内容 

(?P=name)
# 引入name分组匹配到的字符串

\<number>
# 引入编号为number的分组匹配到的字符串

(?<=正则表达式)
# (?<=\d)a   匹配前面是数字的a

(?<!正则表达式)
# （?<!\d）a 匹配前面不是数字的a

(?=正则表达式)
# a(?=\d)  后面是数字的a

(?!正则表达式)
# a(?!\d)  后面不等于数字的a

(?#...)
# 作为注释

(?:正则表达式)
# 同（...）

|
# 先匹配左边表达式，一旦成功就跳过右边表达式


(?iLmsux)


(?(id/name)yes-pattern|no-pattern)
# (\d)abc(?(1)\d|abc) 可匹配至 1abc2 和 abcabc
# 若编号为?(id/name)组匹配到字符串，则执行yes-pattern匹配，否则执行no-pattern匹配

贪婪模式       
ab*  匹配 abbbc

非贪婪模式     
ab*?  匹配 a
```

