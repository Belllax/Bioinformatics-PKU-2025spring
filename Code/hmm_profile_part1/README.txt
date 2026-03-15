这是第二部分的源码，输入及输出结果

本部分包括两个主要代码文件：main.py和ProfileHMM.py。运行程序前，请确保已安装numpy和numba两个依赖库。

所有输入文件和代码文件应位于同一目录下，运行命令如下：

python main.py --train-data train.txt --test-data test.txt --out result.txt

程序运行后将自动生成以下输出：
  •result.txt：预测的序列比对结果
  •hmm_matrices.txt：包含训练得到的 HMM 发射与转移矩阵
  •transitions_heatmap.png及emissions_plot.png：两个对模型参数的可视化图
  （保存在当前目录或指定路径中）


