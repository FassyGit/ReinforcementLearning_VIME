@Lumin
VIME.py里代码开头的一些参数可调
目前存在的问题
USE_VIME = False时的效果明显比True要好，这一点在无论对于intrinsic reward是否进行normalize时都成立（CartPolev-0）
也就是说不加VIME更好，我还没有试其他的GAME
试其他GAME的GYM的环境需要改变
关于Modify 环境和导出视频： https://github.com/openai/gym/wiki/FAQ
----------------------------------------------------------------------------------------------------------
