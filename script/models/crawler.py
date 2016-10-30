import requests #首先导入库
import  re
MaxSearchPage = 20 # 收索页数
CurrentPage = 0 # 当前正在搜索的页数
DefaultPath = "/Users/caishilin/Desktop/pictures" # 默认储存位置
NeedSave = 0 # 是否需要储存
def imageFiler(content): # 通过正则获取当前页面的图片地址数组
    return re.findall('"objURL":"(.*?)"',content,re.S)
def nextSource(content): # 通过正则获取下一页的网址
    next = re.findall('<div id="page">.*<a href="(.*?)" class="n">',content,re.S)[0]
    print("---------" + "http://image.baidu.com" + next)
    return next