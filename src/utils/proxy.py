import os

class ProxyManager():
    """
        需要在服务器上手动配置clash等此类代理软件，运行之后将替换为自己的运行url
    """
    def __init__(self):
        self.http_proxy = 'http://127.0.0.1:7890' # 本地clash默认地址
        self.https_proxy = 'http://127.0.0.1:7890'

    def set_proxy(self):
        """设置代理"""
        os.environ['http_proxy'] = self.http_proxy
        os.environ['https_proxy'] = self.https_proxy

    def unset_proxy(self):
        """取消代理"""
        if 'http_proxy' in os.environ:
            del os.environ['http_proxy']
        if 'https_proxy' in os.environ:
            del os.environ['https_proxy']