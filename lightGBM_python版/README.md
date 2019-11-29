# LightGBM python版
目前网上对于LightGBM的解释仅仅是对论文的翻译，论文中的伪代码到底是怎么用程序实现的，并不了解，所以有种似懂非懂的感觉。官方的源码使用c++写的，看了下，这是个啥？？？？？<br/>
<br/>
这里简要的对lightgbm做了下实现，实际上就是在gbdt的基础上加了点东西。<br/>
EFB：里面的互斥特征绑定实现方式有待改进；<br/>
GOSS：我觉的这个地方没有什么问题；<br/>
直方图：emmm，需要改进啊！<br/>

Reference：<br/>
https://github.com/Microsoft/LightGBM<br/>
http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree<br/>
https://lightgbm.apachecn.org/<br/>
https://zhuanlan.zhihu.com/p/35155992<br/>
https://zhuanlan.zhihu.com/p/85053333<br/>
