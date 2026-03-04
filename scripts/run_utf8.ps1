# 设置 PowerShell 控制台为 UTF-8 编码
# 解决 Windows 下中文日志乱码问题

# 设置控制台输出编码为 UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

# 设置 PowerShell 默认编码
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

# 激活虚拟环境
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"

# 运行 M5 单井训练
& python "$PSScriptRoot\src\run_m5_single_well.py" $args
