from pyecharts.globals import ChartType
from wordcloud import WordCloud, ImageColorGenerator
from pyecharts.charts import Bar, Geo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jieba
import seaborn as sns
from pyecharts import options as opts

# 设置列名与数据对齐
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
# 显示10行
pd.set_option('display.max_rows', 10)
# 读取数据
df = pd.read_csv('subway_all.csv', header=None, names=['city', 'line', 'station'], encoding='gbk')
# 各个城市地铁线路情况
df_line = df.groupby(['city', 'line']).count().reset_index()


def create_map(df):
    # 绘制地图
    value = [i for i in df['line']]
    attr = [i for i in df['city']]
    geo = (
        Geo(init_opts=opts.InitOpts(width='800px', height='400px', bg_color='#404a59'))
        .add_schema(maptype="china")
        .add("已开通地铁城市分布情况", [list(z) for z in zip(attr, value)], type_=ChartType.EFFECT_SCATTER)
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(min_=0, max_=25),
            title_opts=opts.TitleOpts(title="已开通地铁城市分布情况", pos_left="center",
                                      title_textstyle_opts=opts.TextStyleOpts(color="#fff")),
            toolbox_opts=opts.ToolboxOpts(
                feature={
                    "saveAsImage": {},
                    "dataView": {},
                }
            ),
            graphic_opts=[opts.GraphicGroup(
                graphic_item=opts.GraphicItem(
                    left="center", top="bottom", z=100
                ),
                children=[
                    opts.GraphicRect(
                        graphic_item=opts.GraphicItem(left=0, top=0, z=100, width=400, height=50),
                        graphic_shape_opts=opts.GraphicShapeOpts(width=400, height=50),
                        graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(fill="#404a59")
                    ),
                    opts.GraphicText(
                        graphic_item=opts.GraphicItem(left=0, top=0, z=100, width=400, height=50),
                        graphic_textstyle_opts=opts.GraphicTextStyleOpts(
                            text="已开通地铁城市数量",
                            font="20px Microsoft YaHei",
                            graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(fill="#fff")
                        )
                    )
                ]
            )]
        )
    )
    geo.render("C:/Users/72466/Desktop/智慧城市/src/templates/已开通地铁城市分布情况.html")


def create_line(df):
    """
    生成城市地铁线路数量分布情况
    """
    title_len = df['line']
    bins = [0, 5, 10, 15, 20, 25]
    level = ['0-5', '5-10', '10-15', '15-20', '20以上']
    len_stage = pd.cut(title_len, bins=bins, labels=level).value_counts().sort_index()

    attr = len_stage.index.tolist()
    v1 = len_stage.values.tolist()

    bar = Bar()
    bar.add_xaxis(attr)
    bar.add_yaxis("", v1, stack=True, label_opts=opts.LabelOpts(is_show=True))
    bar.set_series_opts(label_opts=opts.LabelOpts(position="inside"))
    bar.set_global_opts(
        title_opts=opts.TitleOpts(title="各城市地铁线路数量分布", pos_left="center",
                                  title_textstyle_opts=opts.TextStyleOpts(color="#000")),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
        yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
        toolbox_opts=opts.ToolboxOpts(
            feature={
                "saveAsImage": {},
                "dataView": {}
            }
        ),
    )

    bar.render("C:/Users/72466/Desktop/智慧城市/src/templates/各城市地铁数量分布.html")


# 各个城市地铁线路数
df_city = df_line.groupby(['city']).count().reset_index().sort_values(by='line', ascending=False)
print(df_city)
create_map(df_city)
create_line(df_city)

# 哪个城市哪条线路地铁站最多
print(df_line.sort_values(by='station', ascending=False))

# 去除重复换乘站的地铁数据
df_station = df.groupby(['city', 'station']).count().reset_index()
print(df_station)

# 统计每个城市包含地铁站数(已去除重复换乘站)
print(df_station.groupby(['city']).count().reset_index().sort_values(by='station', ascending=False))


def create_wordcloud(df):
    """
    生成地铁名词云
    """
    # 分词
    text = ''
    for line in df['station']:
        text += ' '.join(jieba.cut(line, cut_all=False))
        text += ' '
    backgroud_Image = plt.imread('tree2.jpg')
    wc = WordCloud(
        background_color='white',
        mask=backgroud_Image,
        font_path='STXINGKA.TTF',
        max_words=1000,
        max_font_size=150,
        min_font_size=15,
        prefer_horizontal=1,
        random_state=50,
    )
    wc.generate_from_text(text)
    img_colors = ImageColorGenerator(backgroud_Image)
    wc.recolor(color_func=img_colors)
    # 看看词频高的有哪些
    process_word = WordCloud.process_text(wc, text)
    sort = sorted(process_word.items(), key=lambda e: e[1], reverse=True)
    print(sort[:50])
    plt.imshow(wc)
    plt.axis('off')
    wc.to_file("C:/Users/72466/Desktop/智慧城市/src/static/地铁名词云.jpg")
    print('生成词云成功!')


create_wordcloud(df_station)

words = []
for line in df['station']:
    for i in line:
        # 将字符串输出一个个中文
        words.append(i)


def all_np(arr):
    """
    统计单字频率
    """
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result


def create_word(word_message):
    """
    生成柱状图
    """
    attr = [j[0] for j in word_message]
    v1 = [j[1] for j in word_message]
    bar = Bar()
    bar.set_global_opts(
        title_opts=opts.TitleOpts(title="中国地铁站最爱用的字", pos_top='18', pos_left='center'),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
        yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
        toolbox_opts=opts.ToolboxOpts(
            feature={
                "saveAsImage": {},
                "dataView": {},
            }
        )
    )
    bar.add_xaxis(attr)
    bar.add_yaxis("", v1, stack=True, label_opts=opts.LabelOpts(is_show=True))

    bar.render("C:/Users/72466/Desktop/智慧城市/src/templates/中国地铁站最爱用的字.html")


word = all_np(words)
word_message = sorted(word.items(), key=lambda x: x[1], reverse=True)[:10]
create_word(word_message)


def create_door():
    """
    生成柱状图
    """
    attr = ["北京", "西安", "南京"]
    v1 = [35, 13, 13]

    bar = (
        Bar(init_opts=opts.InitOpts(width='800px', height='400px'))
        .add_xaxis(attr)
        .add_yaxis("", v1, stack=True, label_opts=opts.LabelOpts(is_show=True))
        .set_series_opts(label_opts=opts.LabelOpts(position="inside"))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="地铁站最爱用“门”命名的城市", pos_left="center",
                                      title_textstyle_opts=opts.TextStyleOpts(color="#000")),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45)),
            yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
            toolbox_opts=opts.ToolboxOpts(
                feature={
                    "saveAsImage": {},
                    "dataView": {},
                }
            )
        )
    )
    bar.render("C:/Users/72466/Desktop/智慧城市/src/templates/地铁站最爱用门命名的城市.html")


# 选取地铁站名字包含门的数据
df1 = df_station[df_station['station'].str.contains('门')]
# 对数据进行分组计数
create_door()

# 选取数量前5个名字中带有大学的地铁站的城市，并绘制柱状图
df1 = df[df['station'].str.contains('大学')]
city_counts = df1['city'].value_counts()
plt.figure(figsize=(10, 5))
labelline = list(city_counts[:5].index)  #
print(labelline)  # ['上海', '沈阳', '北京', '天津', '重庆']
plt.xlabel('城市')
plt.ylabel('站点数量')
plt.title('名字中带有大学的地铁站的城市数量分布')
plt.bar([i for i in labelline], city_counts[:5])

# 汉字字体，优先使用楷体，找不到则使用黑体
plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']
# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False

plt.savefig('C:/Users/72466/Desktop/智慧城市/src/static/名字中带有大学的地铁站的城市数量分布')

# 绘制北京、武汉、天津、上海等各线路站点数量的折线图趋势分布
# 北京：
df1 = df[df['city'] == '北京']
Bei_station = df1['line'].value_counts()
print(Bei_station)
plt.figure(figsize=(12, 6))
labelline = list(Bei_station[:8].index)
plt.xlabel = ('线路')
plt.ylabel = ('各站点数量')
plt.title("北京各线路站点数量的分布趋势")
plt.plot([i for i in labelline], Bei_station[:8])
plt.savefig('C:/Users/72466/Desktop/智慧城市/src/static/北京各线路站点数量的分布趋势')

# 武汉
df1 = df[df['city'] == '武汉']
Wu_station = df1['line'].value_counts()
print(Wu_station)
plt.figure(figsize=(12, 6))
labelline = list(Wu_station[:8].index)
plt.xlabel = ('线路')
plt.ylabel = ('各站点数量')
plt.title("武汉各线路站点数量的分布趋势")
plt.plot([i for i in labelline], Wu_station[:8])
plt.savefig('C:/Users/72466/Desktop/智慧城市/src/static/武汉各线路站点数量的分布趋势')

# 天津
df1 = df[df['city'] == '天津']
Tian_station = df1['line'].value_counts()
print(Tian_station)
plt.figure(figsize=(12, 6))
labelline = list(Tian_station[:8].index)
plt.xlabel = ('线路')
plt.ylabel = ('各站点数量')
plt.title("天津各线路站点数量的分布趋势")
plt.plot([i for i in labelline], Tian_station[:8])
plt.savefig('C:/Users/72466/Desktop/智慧城市/src/static/天津各线路站点数量的分布趋势')

# 上海
df1 = df[df['city'] == '上海']
Shang_station = df1['line'].value_counts()
print(Shang_station)
plt.figure(figsize=(12, 6))
labelline = list(Shang_station[:8].index)
plt.xlabel = ('线路')
plt.ylabel = ('各站点数量')
plt.title("上海各线路站点数量的分布趋势")
plt.plot([i for i in labelline], Shang_station[:8])
plt.savefig('C:/Users/72466/Desktop/智慧城市/src/static/上海各线路站点数量的分布趋势')

# 哈尔滨
df1 = df[df['city'] == '哈尔滨']
Ha_station = df1['line'].value_counts()
print(Ha_station)
plt.figure(figsize=(12, 6))
labelline = list(Ha_station[:8].index)
plt.xlabel = ('线路')
plt.ylabel = ('各站点数量')
plt.title("哈尔滨各线路站点数量的分布趋势")
plt.plot([i for i in labelline], Ha_station[:8])
plt.savefig('C:/Users/72466/Desktop/智慧城市/src/static/哈尔滨各线路站点数量的分布趋势')

# 各个城市的线路数量的饼状图分布
line_count = df['city'].value_counts().head(25)
labels = [f'{city} ({count}条)' for city, count in zip(line_count.index, line_count.values)]
plt.figure(figsize=(10, 10))  # 增大图形的大小
plt.pie(line_count, labels=labels, autopct='%1.1f%%')
plt.title('各个城市的线路数量的饼状图分布')
plt.savefig(
    'C:/Users/72466/Desktop/智慧城市/src/static/各个城市的线路数量的饼状图分布（25个城市）')

# 各个城市的站点数量的饼状图分布
df_station = df.groupby(['city', 'station']).count().reset_index()
df1 = df_station.groupby(['city']).count().reset_index().sort_values(by='station', ascending=False)
df1['city'] = df1['city'] + '(站点数' + df1['station'].map(str) + ')'
line_count = df1['station'].head(25)
city_labels = df1['city'].head(25)
fig, ax = plt.subplots(figsize=(10, 10))
ax.pie(line_count, labels=city_labels, autopct='%1.1f%%')
ax.set_title('各个城市的站点数量的饼状图分布')
ax.set_aspect('equal')  # 将图形设置为正圆形
plt.savefig(
    'C:/Users/72466/Desktop/智慧城市/src/static/各个城市的站点数量的饼状图分布(25个城市)')

# 各城市的每条线路的站点数量的变化 折线图
df1 = df_line.sort_values(by='station', ascending=False)  # by中指定按照什么列排序，ascending中默认升序排列，值为True
station_count = df1['line'] + df1['city']
plt.figure(figsize=(15, 8))
labelline = list(station_count[:12])
plt.xlabel = ('线路')
plt.ylabel = ('各站点数量')
plt.title("各城市各线路的站点数量前10的变化")
plt.plot([i for i in labelline], df1['station'][:12])
plt.savefig('C:/Users/72466/Desktop/智慧城市/src/static/各城市各线路的站点数量前10的变化')

# 每个城市的哪条线路的地铁站点数量最多  柱形图
df_1 = df_line.sort_values(by='station', ascending=False)
df_2 = df_1.groupby('city')['station'].max().reset_index(drop=False)  # 保留索引
line_station_c = df_2.sort_values(by='station', ascending=False)
plt.figure(figsize=(15, 5))
labelline = list(line_station_c['city'])

labelline = labelline  # +line_text['line'].map(str)
plt.xlabel = ('城市')
plt.ylabel = ('站点数量')
plt.bar([i for i in labelline], line_station_c['station'])
plt.title('各个城市地铁线路的最大站点数')
plt.savefig('C:/Users/72466/Desktop/智慧城市/src/static/各个城市地铁线路的最大站点数')

# 统计各个城市的大学数量，然后利用回归图进行拟合（分析各个城市的大学数量与站点数量的关系
df_uni = pd.read_csv('./university.csv', header=None, names=['city', 'uni_count'], encoding='gbk')
df_uni = pd.merge(left=line_station_c, right=df_uni, on='city', how='inner')  # 将两个表格中的数据基于city列进行内连接。
x = df_uni['uni_count']
y = df_uni['station']
sns.regplot(x=x, y=y, color='b')
plt.title('分析各个城市的大学数量与站点数量的关系')
plt.savefig('C:/Users/72466/Desktop/智慧城市/src/static/分析各个城市的大学数量与站点数量的关系')

# 散点图
fig = plt.figure(figsize=(10, 7))
plt.xlabel = ('站点数量')
plt.ylabel = ('大学数量')
plt.title('各个城市的大学数量与站点数量的关系')
plt.scatter(x=x, y=y, color='blue', marker='*', alpha=0.8)
plt.grid()
plt.savefig(
    'C:/Users/72466/Desktop/智慧城市/src/static/各个城市的大学数量与站点数量的关系')  # plt.show()

df_s = df_uni
sns.jointplot(x='uni_count', y='station', data=df_s)
plt.close()

# 选取郑州、武汉、广州、长沙同名的线路1-线路6，绘制折线图分析这些城市的目标线路的站点数量分布
df_1 = df_line.sort_values(by='station', ascending=False)
zz_ = df_1[df_1['city'] == '郑州'].sort_values(by='line',
                                               ascending=False).reset_index()
zz_ = zz_.loc[zz_['line'].isin(['1号线', '2号线', '3号线', '4号线', '5号线', '6号线'])]

wh_ = df_1[df_1['city'] == '武汉'].sort_values(by='line', ascending=False).reset_index()
wh_ = wh_.loc[wh_['line'].isin(['1号线', '2号线', '3号线', '4号线', '5号线', '6号线'])]

gz_ = df_1[df_1['city'] == '广州'].sort_values(by='line', ascending=False).reset_index()
gz_ = gz_.loc[gz_['line'].isin(['1号线', '2号线', '3号线', '4号线', '5号线', '6号线'])]

cs_ = df_1[df_1['city'] == '长沙'].sort_values(by='line', ascending=False).reset_index()
cs_ = cs_.loc[cs_['line'].isin(['1号线', '2号线', '3号线', '4号线', '5号线', '6号线'])]

plt.figure(figsize=(10, 7))
L1 = plt.plot(zz_['line'], zz_['station'], color='b', label='郑州线路1-6的站点数量变化')
L2 = plt.plot(wh_['line'], wh_['station'], color='g', label='武汉线路1-6的站点数量变化')
L3 = plt.plot(gz_['line'], gz_['station'], color='r', label='广州线路1-6的站点数量变化')
L4 = plt.plot(cs_['line'], cs_['station'], color='k', label='长沙线路1-6的站点数量变化')
plt.legend()
plt.title('郑州、武汉、广州、长沙同名的线路1-线路6的站点数量分布')
plt.xlabel = ('线路1-线路6')
plt.ylabel = ('站点数量')
plt.savefig(
    'C:/Users/72466/Desktop/智慧城市/src/static/郑州、武汉、广州、长沙同名的线路1-线路6的站点数量分布')

# 选取广州、天津、武汉、重庆同名的线路1-线路6，绘制折线图分析这些城市的目标线路的站点数量分布
df_1 = df_line.sort_values(by='station', ascending=False)
zz_ = df_1[df_1['city'] == '广州'].sort_values(by='line',
                                               ascending=False).reset_index()
zz_ = zz_.loc[zz_['line'].isin(['1号线', '2号线', '3号线', '4号线', '5号线', '6号线'])]

wh_ = df_1[df_1['city'] == '天津'].sort_values(by='line', ascending=False).reset_index()
wh_ = wh_.loc[wh_['line'].isin(['1号线', '2号线', '3号线', '4号线', '5号线', '6号线'])]

gz_ = df_1[df_1['city'] == '武汉'].sort_values(by='line', ascending=False).reset_index()
gz_ = gz_.loc[gz_['line'].isin(['1号线', '2号线', '3号线', '4号线', '5号线', '6号线'])]

cs_ = df_1[df_1['city'] == '重庆'].sort_values(by='line', ascending=False).reset_index()
cs_ = cs_.loc[cs_['line'].isin(['1号线', '2号线', '3号线', '4号线', '5号线', '6号线'])]

plt.figure(figsize=(10, 7))
L1 = plt.plot(zz_['line'], zz_['station'], color='b', label='广州线路1-6的站点数量变化')
L2 = plt.plot(wh_['line'], wh_['station'], color='g', label='天津线路1-6的站点数量变化')
L3 = plt.plot(gz_['line'], gz_['station'], color='r', label='武汉线路1-6的站点数量变化')
L4 = plt.plot(cs_['line'], cs_['station'], color='k', label='重庆线路1-6的站点数量变化')
plt.legend()
plt.title('广州、天津、武汉、重庆同名的线路1-线路6的站点数量分布')
plt.xlabel = ('线路1-线路6')
plt.ylabel = ('站点数量')
plt.savefig(
    'C:/Users/72466/Desktop/智慧城市/src/static/广州、天津、武汉、重庆同名的线路1-线路6的站点数量分布')

# 全国各城市的总的换乘站点数量（2换乘、3换乘、4换乘等）分布统计
df_1 = df.groupby(['city', 'station']).count().reset_index()
print(df_1)
df_1 = df_1[df_1['line'] > 1]  # 筛选出来全国的换乘站点数
tran_sit = df_1.groupby('line').count().reset_index()  # 保留原索引，但是值是count()函数计数之后的值
plt.figure(figsize=(10, 5))
plt.xlabel = ('站点可换乘等级')
plt.ylabel = ('站点数量')
plt.bar(tran_sit['line'], tran_sit['station'], color='g')
plt.title('全国各城市总的换乘站点数量（2换乘、3换乘、4换乘等）分布统计')
plt.savefig(
    'C:/Users/72466/Desktop/智慧城市/src/static/全国各城市总的换乘站点数量（2换乘、3换乘、4换乘等）分布统计')
print(tran_sit[tran_sit['line'] == 5]['station'])
