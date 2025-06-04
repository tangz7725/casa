# Casanovo

#### 创建conda环境



```python
conda create --name casa python=3.10
```

```
conda activate casa
```

#### 安装Casanovo

```
pip install casanovo
```

查看是否安装成功：

```
casanovo --help
```

所有配置可在.yaml文件指定，使用如下命令生成默认.yaml文件

```
casanovo configure
```

使用 `casanovo sequence` 命令对质谱数据测序：

```
python.exe -m casanovo.casanovo sequence -o results.mztab spectra.mgf
```

从头开始训练模型则使用 `train` 命令

```
python.exe -m casanovo.casanovo train --validation_peak_path validation_spectra.mgf training_spectra.mgf
```

使用`-g`参数可指定faa组学数据，如：

```
python.exe -m casanovo.casanovo sequence your.mgf -g your.faa -o your.mztab
```
