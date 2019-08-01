# Statistical Graph

1.茎叶图

```python
from itertools import groupby
data = '89 79 57 46 1 24 71 5 6 9 10 15 16 19 22 31 40 41 52 55 60 61 65 69 70 75 85 91 92 94'

for k, g in groupby(sorted(data.split()), key=lambda x: int(x) // 10):
    lst = map(str, [int(_) % 10 for _ in list(g)])
    print(k, '|', ' '.join(lst))
```
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
```
#Matplotlib

Matplotlib是一种2D的绘图库，它可以支持硬拷贝和跨系统的交互，它可以在Python脚本、IPython的交互环境下、Web应用程序中使用。该项目是由John Hunter于2002年启动的，其目的是为Python构建一个MATLAB式的绘图接口。如果结合使用一种GUI工具包（如IPython），Matplotlib还具有诸如缩放和平移等交互功能。它不仅支持各种操作系统上许多不同的GUI后端，而且还能将图片导出为各种常见的食量（vector）和光栅（raster）图：PDF、SVG、JPG、PNG、BMP、GIF等。

**Matplotlib程序包**

 所谓“一图胜千言”，我们很多时候需要通过可视化的方式查看、分析数据，虽然Pandas中也有一些绘图操作，但是相比较而言，Matplotlib在绘图显示效果方面更加出色。Python为Matplotlib提供了一个方便的接口，我们可以通过Pyplot对Matplotlib进行操作，多数情况下，Pyplot的命令与MATLAB有些相似。

导入Matplotlib包进行简单的操作（此处需要安装pip install matplotlib）：

```python
import matplotlib.pyplot as plt#约定俗成的写法plt
#首先定义两个函数（正弦&余弦）
import numpy as np

X=np.linspace(-np.pi,np.pi,256,endpoint=True)#-π to+π的256个值
C,S=np.cos(X),np.sin(X)
plt.plot(X,C)
plt.plot(X,S)
#在ipython的交互环境中需要这句话才能显示出来
plt.show()
```

输出结果：

![image_1dh7iin1p18qs1cbt17l71km10q9.png-18.3kB][1]
 
##绘图命令的基本架构及其属性设置

 上面的例子我们可以看出，几乎所有的属性和绘图的框架我们都选用默认设置。现在我们来看Pyplot绘图的基本框架是什么，用过Photoshop的人都知道，作图时先要定义一个画布，此处的画布就是Figure，然后再把其他素材“画”到该Figure上。

**1）在Figure上创建子plot，并设置属性**

```python
x=np.linspace(0,10,1000)#X轴数据
y1=np.sin(x)#Y轴数据
y2=np.cos(x**2)#Y轴数据  x**2即x的平方

plt.figure(figsize=(8,4))

plt.plot(x,y1,label="$sin(x)$",color="red",linewidth=2)#将$包围的内容渲染为数学公式
plt.plot(x,y2,"b--",label="$cos(x^2)$")
#指定曲线的颜色和线性，如‘b--’表示蓝色虚线（b：蓝色，-：虚线）

plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("PyPlot First Example")

'''
使用关键字参数可以指定所绘制的曲线的各种属性：
label：给曲线指定一个标签名称，此标签将在图标中显示。如果标签字符串的前后都有字符'$'，则Matplotlib会使用其内嵌的LaTex引擎将其显示为数学公式
color：指定曲线的颜色。颜色可以用如下方法表示
       英文单词
       以‘#’字符开头的3个16进制数，如‘#ff0000’表示红色。
       以0~1的RGB表示，如（1.0,0.0,0.0）也表示红色。
linewidth：指定权限的宽度，可以不是整数，也可以使用缩写形式的参数名lw。
'''

plt.ylim(-1.5,1.5)
plt.legend()#显示左下角的图例

plt.show()
```
![image_1dh7illugmlt1i2gke31saj1g4lm.png-43.5kB][2]

**2）在Figure上创建多个子plot**

如果需要绘制多幅图表的话，可以给Figure传递一个整数参数指定图表的序号，如果所指定序号的绘图对象已经存在的话，将不创建新的对象，而只是让它成为当前绘图对象。

```python
fig1=plt.figure(2)
plt.subplot(211)
#subplot(211)把绘图区域等分为2行*1列共两个区域，然后在区域1（上区域）中创建一个轴对象
plt.subplot(212)#在区域2（下区域）创建一个轴对象
plt.show()
```

输出结果：

![image_1dh7inf95oo1j88p5d19c69dk13.png-7.1kB][3]

我们还可以通过命令再次拆分这些块（相当于Word中拆分单元格操作）

```python
f1=plt.figure(5)#弹出对话框时的标题，如果显示的形式为弹出对话框的话
plt.subplot(221)
plt.subplot(222)
plt.subplot(212)
plt.subplots_adjust(left=0.08,right=0.95,wspace=0.25,hspace=0.45)
# subplots_adjust的操作时类似于网页css格式化中的边距处理，左边距离多少？
# 右边距离多少？这取决于你需要绘制的大小和各个模块之间的间距
plt.show()
```

输出结果：

![image_1dh7irjp7qii13l21ua91089b3j1g.png-7.8kB][4]

**3）通过Axes设置当前对象plot的属性**

 以上我们操作的是在Figure上绘制图案，但是当我们绘制图案过多，又需要选取不同的小模块进行格式化设置时，Axes对象就能很好地解决这个问题。

```python
fig,axes=plt.subplots(nrows=2,ncols=2)#定一个2*2的plot
plt.show()
```

输出结果：

![image_1dh7itatb1urhoqq14eo74i18a91t.png-8.2kB][5]

现在我们需要通过命令来操作每个plot（subplot），设置它们的title并删除横纵坐标值。

```python
fig,axes=plt.subplots(nrows=2,ncols=2)#定一个2*2的plot
axes[0,0].set(title='Upper Left')
axes[0,1].set(title='Upper Right')
axes[1,0].set(title='Lower Left')
axes[1,1].set(title='Lower Right')

# 通过Axes的flat属性进行遍历
for ax in axes.flat:
#     xticks和yticks设置为空置
    ax.set(xticks=[],yticks=[])
plt.show()
```

输出结果：

![image_1dh7iuos911fqt4g16lr16p15ef3a.png-4.8kB][6]

另外，实际来说，plot操作的底层操作就是Axes对象的操作，只不过如果我们不使用Axes而用plot操作时，它默认的是plot.subplot(111)，也就是说plot其实是Axes的特例。

 **4）保存Figure对象**

最后一项操作就是保存，我们绘图的目的是用在其他研究中，或者希望可以把研究结果保存下来，此时需要的操作时save。

```python
plt.savefig(r"C:\Users\123\Desktop\save_test.png",dpi=520)#默认像素dpi是80
```

很明显保存的像素越高，内存越大。此处只是用了savefig属性对Figure进行保存。

另外，除了上述的基本操作之外，Matplotlib还有其他的绘图优势，此处只是简单介绍了它在绘图时所需要注意的事项，更多的属性设置请参考：https://matplotlib.org/api/

#Seaborn模块介绍

前面我们简单介绍了Matplotlib库的绘图功能和属性设置，对于常规性的绘图，使用Pandas的绘图功能已经足够了，但如果对Matplotlib的API属性研究较为透彻，几乎没有不能解决的问题。但是Matplotlib还是有它的不足之处，Matplotlib自动化程度非常高，但是，掌握如何设置系统以便获得一个吸引人的图是相当困难的事。为了控制Matplotlib图表的外观，Seaborn模块自带许多定制的主题和高级的接口。

**1）未加Seaborn模块的效果**

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(sum(map(ord,"aesthetics")))
#首先定义一个函数用来画正弦函数，可帮助了解可以控制的不同风格参数
def sinplot(flip=1):
    x=np.linspace(0,14,100)
    for i in range(1,7):
        plt.plot(x,np.sin(x+i*0.5)*(7-i)*flip)
sinplot()
plt.show()
```

输出结果：

![image_1dh7j26ee1hkd1tmrs97135c168847.png-42.7kB][7]

**2）加入Seaborn模块的效果**

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 添加了Seaborn模块

np.random.seed(sum(map(ord,"aesthetics")))
#首先定义一个函数用来画正弦函数，可帮助了解可以控制的不同风格参数
def sinplot(flip=1):
    x=np.linspace(0,14,100)
    for i in range(1,7):
        plt.plot(x,np.sin(x+i*0.5)*(7-i)*flip)
#转换成Seaborn模块，只需要引入seaborn模块
import seaborn as sns#添加Seaborn模块
sinplot()
plt.show()
```

输出效果：

![image_1dh7j3nn91im1coa1ogj1al11mud4k.png-42.7kB][8]

小编使用的jupyter notebook编辑器，使用与不使用Seaborn模块效果差别不明显。

使用Seaborn的优点有：

- Seaborn默认浅灰色背景与白色网格线的灵感来源于Matplotlib，却比Matplotlib的颜色更加柔和
- Seaborn把绘图风格参数与数据参数分开设置。

其中，Seaborn有两组函数对风格进行控制：axes_style()/set_style()函数和plotting_context()/set_context()函数。

axes_style()函数和plotting_context()函数返回参数字典，set_style()函数和set_context()函数设置Matplotlib。

**使用set_style()函数**

```python
import seaborn as sns

'''
Seaborn有5种预定义的主题：
darkgrid（灰色背景+白网格）
whitegrid（白色背景+黑网格）
dark（仅灰色背景）
white（仅白色背景）
ticks（坐标轴带刻度）
默认的主题是darkgrid，修改主题可以使用set_style函数
'''
sns.set_style("whitegrid")
sinplot()#即上段代码中定义的函数
plt.show()
```

输出结果：

![image_1dh7j5q2l1m1o14c01na74k1gbu51.png-44.2kB][9]

**使用set_context()函数**

```python
'''
上下文（context）可以设置输出图片的大小尺寸（scale）
Seaborn中预定义的上下文有4种：paper、notebook、talk和poster
默认使用notebook上下文
'''
sns.set_context("poster")
sinplot()#即前文定义的函数
plt.show()
```

输出结果：

![image_1dh7j7e84vh61fskclq1n511o915e.png-49.2kB][10]

**使用Seaborn“耍酷”**

 然而Seaborn不仅能够用来更改背景颜色，或者改变画布大小，还有其他很多方面的用途，比如下面的例子。

```python
'''
Annotated heatmaps
================================
'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#通过加载sns自带数据库中的数据（具体数据可以不关心）
flights_long=sns.load_dataset("flights")
flights=flights_long.pivot("month","year","passengers")

# 使用每个单元格中的数据值绘制一个热力图heatmap
sns.heatmap(flights,annot=True,fmt="d",linewidths=.5)
plt.show()
```

输出结果：

![image_1dh7jc88ngai11671jv71qhm1pmj5r.png-69kB][11]

# 描述性统计图形概览

 描述性统计是借助图表或者总结性的数值来描述数据的统计手段。数据挖掘工作的数据分析阶段，我们可借助描述性统计来描绘或总结数据的基本情况，一来可以梳理自己的思维，而来可以更好地向他人展示数据分析结果。数值分析的过程中，我们往往要计算出数据的统计特征，用来做科学计算的Numpy和Scipy工具可以满足我们的需求。Matplotlib工具可用来绘制图，满足图分析的需求。

**1）制作数据**

数据是自己制作的，主要包括个人身高、体重及一年的借阅图书量（之所以自己制作数据是因为不是每份真实的数据都可以进行接下来的分析，比如有些数据就不能绘制饼图，另一个角度也说明，此处举例的数据其实没有实际意义，只是为了分析二举例，但是不代表在具体的应用中这些分析不能发挥作用）。

另外，以下的数据显示都是在Seaborn库的作用下体现的效果。

```python
# 案例分析
from numpy import array
from numpy.random import normal

def getData():
    heights=[]
    weights=[]
    books=[]
    N=10000
    for i in range(N):
        while True:
            #身高服从均值为172，标准差为6的正态分布
            height=normal(172,6)
            if 0<height:
                break
        while True:
            #体重由身高作为自变量的线性回归模型产生，误差服从标准正态分布
            weight=(height-80)*0.7+normal(0,1)
            if 0<weight:
                break
        while True:
            #借阅量服从均值为20，标准差为5的正态分布
            number=normal(20,5)
            if 0<=number and number<=50:
                book='E' if number<10 else ('D' if number<15 else ('C' if number<20 else ('B' if number<25 else 'A')))
                break
        heights.append(height)
        weights.append(weight)
        books.append(book)
   return array(heights),array(weights),array(books)
heights,weights,books=getData()
```

**2)频数分析**

**（1）定性分析**

柱状图和饼形图是对定性数据进行频数分析的常用工具，使用前需将每一类的频数计算出来。

柱状图。柱状图是以柱的高度来指代某类型的频数，使用Matplotlib对图书借阅量这一定性变量绘制柱状图的代码如下。（接上段代码）

```python
from matplotlib import pyplot

#绘制柱状图
def drawBar(books):
    xticks=['A','B','C','D','E']
    bookGroup={}
    #对每一类借阅量进行频数统计
    for book in books:
        bookGroup[book]=bookGroup.get(book,0)+1
    #创建柱状图
    #第一个参数为柱的横坐标
    #第二个参数为柱的高度
    #参数align为柱的对齐方式，以第一个参数为参考标准
    pyplot.bar(range(5),[bookGroup.get(xtick,0) for xtick in xticks],align='center')
    
    #设置柱的文字说明
    #第一个参数为文字说明的横坐标
    #第二个参数为文字说明的内容
    pyplot.xticks(range(5),xticks)
    #设置横坐标的文字说明
    pyplot.xlabel("Types of Students")
    #设置纵坐标的文字说明
    pyplot.ylabel("Frequency")
    #设置标题
    pyplot.title("Numbers of Books Students Read")
    #绘图
    pyplot.show()
drawBar(books)
```

输出结果：

![image_1dh7jm0gi4sdatj1fiakhv1hbb68.png-13kB][12]

饼形图。饼形图事宜扇形的面积来指代某类型的频率，使用Matplotlib对图书借阅量这一定性变量绘制饼形图的代码如下：

```python
#绘制饼形图
def drawPie(books):
    labels=['A','B','C','D','E']
    bookGroup={}
    for book in books:
        bookGroup[book]=bookGroup.get(book,0)+1
    #创建饼形图
    #第一个参数是扇形的面积
    #labels参数为扇形的说明文字
    #autopct参数为扇形占比的显示格式
    pyplot.pie([bookGroup.get(label,0) for label in labels],labels=labels,autopct='%1.1f%%')
    pyplot.title("Number of Books Students Read")
    pyplot.show()
drawPie(books)
```

输出结果：

![image_1dh7jnt1v1r84tbflql1p6k1acb6l.png-15.3kB][13]

**（2）定量分析**

直方图类似于柱状图，是用柱的高度来指代频数，不同的是其将定量数据划分为若干连续的区间，在这些连续的区间上绘制柱。

直方图。使用Matplotlib对身高这一定量变量绘制直方图的代码如下：

```python
#绘制直方图
def drawHist(heights):
    #创建直方图
    #第一个参数为待绘制的定量数据，不同于定性数据，这里并没有实现进行频数统计
    #第二个参数为划分的区间个数
    pyplot.hist(heights,100)
    pyplot.xlabel('Heights')
    pyplot.ylabel('Frequency')
    pyplot.title('Height of Students')
    pyplot.show()
drawHist(heights)
```

输出结果：

![image_1dh7jp7pdlii1hdm1jvime1gdp72.png-12.4kB][14]

累积曲线。使用Matplotlib对身高这一定量变量绘制累积曲线的代码如下：

```python
#绘制累积曲线
def drawCumulativaHist(heights):
    #创建累积曲线
    #第一个参数为待绘制的定量数据
    #第二个参数为划分的区间个数
    #normal参数为是否无量纲化
    #histtype参数为‘step’，绘制阶梯状的曲线
    #cumulative参数为是否累积
    pyplot.hist(heights,20,normed=True,histtype='step',cumulative=True)
    pyplot.xlabel('Heights')
    pyplot.ylabel('Frequency')
    pyplot.title('Heights of Students')
    pyplot.show()
drawCumulativaHist(heights)
```

输出结果：

![image_1dh7jq2sp1gisa331q3g1kd91knt7v.png-10.3kB][15]

**3)关系分析**

 散点图。在散点图中，分别以自变量和因变量作为横坐标。当自变量与因变量线性相关时，散点图中的点近似分布在一条直线上。我们以身高作为自变量，体重作为因变量，讨论身高对体重的影响。使用Matplotlib绘制散点图的代码如下：

```python
#绘制散点图
def drawScatter(heights,weights):
    #创建散点图
    #第一个参数为点的横坐标
    #第二个参数为点的纵坐标
    pyplot.scatter(heights,weights)
    pyplot.xlabel('Heights')
    pyplot.ylabel('Weight')
    pyplot.title('Heights & Weight of Students')
    pyplot.show()
drawScatter(heights,weights)
```

输出结果:

![image_1dh7jrf9s12kb1qn81j1q8vq2cs8c.png-13.5kB][16]

**4)探索分析**

 箱型图。在不明确数据分析的目标时，我们对数据进行一些探索性的分析，可以知道数据的中心位置、发散程度及偏差程度。使用Matplotlib绘制关于身高的箱型图代码如下：

```python
#绘制箱型图
def drawBox(heights):
    #创建箱型图
    #第一个参数为待绘制的定量数据
    #第二个参数为数据的文字说明
    pyplot.boxplot([heights],labels=['Heights'])
    pyplot.title('Heights of Students')
    pyplot.show()
drawBox(heights)
```

输出结果：

![image_1dh7jshss1gvub2719jd18ri15658p.png-9.1kB][17]

**注：**

上四分位数与下四分位数的差叫四分位差，它是衡量数据发散程度的指标之一
上界线和下界线是距离中位数1.5倍四分位差的线，高于上界线或者低于下界线的数据为异常值
 描述性统计是容易操作、直观简洁的数据分析手段。但是由于简单，对于多元变量的关系难以描述。现实生活中，自变量通常是多元的：决定体重的不仅有身高，还有饮食习惯、肥胖基因等因素。通过一些高级的数据处理手段，我们可以对多元变量进行处理，例如，特征工程中，可以使用互信息方法来选择多个对因变量有较强相关性的自变量作为特征，还可以使用主成分分析法来消除一些冗余的自变量来降低运算复杂度。

 

 参考书目：《数据馆员的Python简明手册》


  [1]: http://static.zybuluo.com/harrytsz/veggs5tzq5kqex3u81sixkhu/image_1dh7iin1p18qs1cbt17l71km10q9.png
  [2]: http://static.zybuluo.com/harrytsz/pndgcc39q0y5af1wutb28aw2/image_1dh7illugmlt1i2gke31saj1g4lm.png
  [3]: http://static.zybuluo.com/harrytsz/oll19lg0b4omn5wkcjq6mdo5/image_1dh7inf95oo1j88p5d19c69dk13.png
  [4]: http://static.zybuluo.com/harrytsz/vu385vimm6h5j7zzwp6jdvc6/image_1dh7irjp7qii13l21ua91089b3j1g.png
  [5]: http://static.zybuluo.com/harrytsz/ydmpzcnkyw3vfeg3y0cbci07/image_1dh7itatb1urhoqq14eo74i18a91t.png
  [6]: http://static.zybuluo.com/harrytsz/wr7ebfx1341zmyym7tx4vhoj/image_1dh7iuos911fqt4g16lr16p15ef3a.png
  [7]: http://static.zybuluo.com/harrytsz/e7p4956sxxdoowt7khy1qrpd/image_1dh7j26ee1hkd1tmrs97135c168847.png
  [8]: http://static.zybuluo.com/harrytsz/jgej2s7u7k9vbetfpock1si4/image_1dh7j3nn91im1coa1ogj1al11mud4k.png
  [9]: http://static.zybuluo.com/harrytsz/e170swi3so2wwfsuuzeqyvy1/image_1dh7j5q2l1m1o14c01na74k1gbu51.png
  [10]: http://static.zybuluo.com/harrytsz/v47y7nd2zoyzbvc9up8z32yw/image_1dh7j7e84vh61fskclq1n511o915e.png
  [11]: http://static.zybuluo.com/harrytsz/m79ee2vby5q9ccabt7uag1rn/image_1dh7jc88ngai11671jv71qhm1pmj5r.png
  [12]: http://static.zybuluo.com/harrytsz/mu292oboomv24wagb8854r2y/image_1dh7jm0gi4sdatj1fiakhv1hbb68.png
  [13]: http://static.zybuluo.com/harrytsz/xyh7sw31a1op3o9iio2cbh9j/image_1dh7jnt1v1r84tbflql1p6k1acb6l.png
  [14]: http://static.zybuluo.com/harrytsz/5r7a8ex5pj70pbxy5fl5r6z3/image_1dh7jp7pdlii1hdm1jvime1gdp72.png
  [15]: http://static.zybuluo.com/harrytsz/lqxoxfsbwq47pdqkbwikz5rd/image_1dh7jq2sp1gisa331q3g1kd91knt7v.png
  [16]: http://static.zybuluo.com/harrytsz/2j6ze8xz6t5lxtxv6av538te/image_1dh7jrf9s12kb1qn81j1q8vq2cs8c.png
  [17]: http://static.zybuluo.com/harrytsz/aoor1ds7ur4rqpqsol8x3gwb/image_1dh7jshss1gvub2719jd18ri15658p.png