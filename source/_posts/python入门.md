---
layout: python简介
title: python入门
date: 2019-01-13 19:40:56
categories: python编码
---

### Python：一种交互式、解释性、面向对象的编程语言

解释性语言：即无需编译源码为可执行文件，直接使用源码就可以运行

#### Python shell

提供一个运行环境，方便交互式开发。在windows下，分为两种，Python(command line) 和 IDLE

#### Python IDE

将Python开发相关的各种工具集成，包括python代码编辑器，python运行环境（pyhton shell）。常见主流的IDE有Pycharm和spyder

#### python编码

- **字节与字符**

```
计算机存储的一切数据，文本字符、图片、视频、音频、软件都是由一串01的字节序列构成的，一个字节等于8个比特位。

而字符就是一个符号，比如一个汉字、一个英文字母、一个数字、一个标点都可以称为一个字符。

字节方便存储和网络传输，而字符用于显示，方便阅读。例如字符 "p" 存储到硬盘是一串二进制数据 01110000，占用一个字节的长度
```

- **len\(\) 函数** 

```
len(string)返回的是字节数，len(unicode)返回的是字符数

gbk编码每个汉字占用2个字节，utf8编码的每个汉字占用3个字节,而1个英文字符只占用1个字节
```

- **编码与解码**

```
我们用编辑器打开的文本，看到的一个个字符，最终保存在磁盘的时候都是以二进制字节序列形式存起来的。那么从字符到字节的转换过程就叫做编码（encode），反过来叫做解码（decode），两者是一个可逆的过程。编码是为了存储传输，解码是为了方便显示阅读
```

- **ascill编码**   

英文字母加上特殊字符，一共128个字符。这个就是ascill编码。对应关系很简单，一个字符对应一个byte

- **Unicode编码**

Unicode（统一码，万国码）是基于通用字符集（Universal Character Set）的标准发展。它为每种语言中的每个字符设定了统一并且唯一的二进制编码，以满足语言、跨平台进行文本转换、处理的要求

- **UTF-8**

Unicode是一个标准，Unicode只是一个符号集，它只规定了符号的二进制代码，却没有规定这个二进制代码应该如何存储；UTF-8是实现，UTF-8以字节为单位对Unicode进行编码  
UTF-8（8-bit Unicode Transformation Format）是一种针对Unicode的可变长度字符编码（定长码），也是一种前缀码。它可以用来表示 Unicode 标准中的任何字符，且其编码中的第一个字节仍与[ASCII]()兼容  
windows记事本保存的utf-8格式，开始的三个字节EF BB BF就是BOM了，全称Byte Order Mark，这玩意也是很多乱码问题的来源，Linux下很多程序就不认BOM。因此，强烈不建议使用BOM，使用Notepad++之类的软件保存文本时，尽量选择以UTF-8无BOM格式编码

- **linux 转UTF-8\(无bom\)**

用 Vim 打开

```
:set nobomb
:wq
```

------

- **Python2**

Python默认脚本文件都是ASCII编码的，当文件中有非ASCII编码范围内的字符的时候就要使用“编码指示”来修正，也就是在文件第一行或第二行指定编码声明：  
`# -*- coding=utf-8 -*`-或者`#coding=utf-8`，若头部声明coding=utf-8, a = '中文' 其编码为utf-8，若头部声明coding=gb2312, a = '中文' 其编码为gbk

python解释器纯粹把源码使用ascii编码进行解析生成语法树。考虑到源码里可能存在其他语言的字符串量，提供了setdefaultencode接口，但是非常容易引发各类问题。PEP263指出在文件第一行或者第二行（仅限第一行为Unix脚本标注的情况下）写入特殊格式的注释\# coding:xxx可以指定解释器解释源码时使用的字符编码

Python2 把字符串分为 unicode 和 str 两种类型。本质上 str 是一串二进制字节序列，我们要把 unicode 符号保存到文件或者传输到网络就需要经过编码处理转换成 str 类型

**python2的print的实质是将str里的东西输出到PIPE，如果你print的是一个unicode对象，它会自动根据LOCALE环境变量进行encode之后变成str再输出。然而一般在Windows上都没有设置locale环境变量，py2就按照默认的ascii编码进行处理，于是对中文自然就编码错误了**。解决方法是手动encode成对应的输出端可接受的编码后输出。win下一般都是gbk，linux下一般都是utf8

------

- **str和unicode对象的转换，通过encode和decode实现**

str对象,存储 bytes，它仅仅是一个字节流，没有其它的含义，如果你想使这个字节流显示的内容有意义，就必须用正确的编码格式，解码显示

如果你使用一个 “u” 前缀，那么你会有一个 “unicode” 对象，存储的是 code points ，在一个 unicode 字符串中，你可以使用反斜杠 u\(u\) 来插入任何的 unicode 代码点

Unicode 字符串会有一个 .encode 方法来产生 bytes , bytes 串会有一个 .decode 方法来产生 unicode 。每个方法中都有一个参数来表明你要操作的编码类型

**decode方法是将一个str按照指定编码解析后转换为unicode，encode方法则是把一个unicode对象用指定编码表示并存储到一个str对象里**

- **字符串前加u如何理解：**

```
两种字符串如何相互转换？字符串'xxx'虽然是ASCII编码，但也可以看成是UTF-8编码，而u'xxx'则只能是Unicode编码
```

- **如何获得系统默认编码**

```
import sys
print sys.getdefaultencoding()
```

- **Python中用encoding声明的文件编码和文件的实际编码之间**

```
关键是保持 编码和解码时用的编码类型一致

>>>x=u"禅"
>>>a=x.encode("utf-8")
>>>a
'\xe7\xa6\x85'
>>>a.decode("gbk")
出现错误，需用decode("utf-8")
```

- **开头声明编码格式的解释**

```
1.如果没有此文件编码类型的声明，则python默认以ASCII编码去处理

2.必须放在python文件的第一行或第二行

3.程序会通过头部声明，解码初始化 u”字符串”，这样的unicode对象，所以头部声明和代码的存储格式要一致
```

```
str = u.encode('utf-8') 
#以utf-8编码对unicode对象进行编码

u = str.decode('gb2312')
#以gb2312编码对字符串str进行解码，以获取unicode

```

eg.

```
>>>su=u'哈哈'
>>>type(su)
<type 'unicode'>
>>>s_utf8=su.encode('utf-8')
>>>type(s_utf8)
<type 'str'>
>>>s_utf8
'\xe5\x93\x88\xe5\x93\x88'
```

- **当将一个unicode对象传给print时，在内部会将该unicode对象进行一次转换，转换成本地的默认编码**

```
>>>print s_utf8
乱码显示
>>>print su
哈哈
>>>s_gbk=su.encode('gbk')
>>>s_gbk
'\xb9\xfe\xb9\xfe'
>>> print s_gbk
哈哈
```



