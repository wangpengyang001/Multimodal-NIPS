# Multimodal-NIPS 安全测试报告

- 测试对象：`wangpengyang001/Multimodal-NIPS`
- 本地测试路径：`Multimodal-NIPS-worktree`
- 测试日期：2026-06-04
- 测试方式：本地静态安全测试、依赖清单审查、Python 语法编译检查
- 测试边界：未对第三方网络目标、真实生产流量或外部主机执行攻击性测试；未执行真实防火墙封禁命令。

## 测试结论

本次测试未发现硬编码口令、API Token、`eval/exec` 动态执行、`pickle` 直接使用等明显恶意代码或后门痕迹。项目通过 Python 语法编译检查。

需要整改或在上线前重点复核的风险主要集中在两类：一是防火墙封禁命令通过 `shell=True` 执行，建议改为参数列表并校验 IP；二是模型文件通过 `torch.load` 加载，建议只加载可信模型并增加完整性校验。

## 测试项目与结果

| 序号 | 测试项 | 方法 | 结果 | 说明 |
|---:|---|---|---|---|
| 1 | 源代码清单核对 | 遍历仓库文件，排除 `__pycache__` | 通过 | 识别 Python 源文件 9 个，非注释代码约 549 行。 |
| 2 | 语法完整性检查 | 执行 `compileall` | 通过 | 未发现 Python 语法错误。 |
| 3 | 硬编码敏感信息扫描 | 扫描 password、secret、token、api_key 等关键词 | 通过 | 未发现疑似硬编码凭据。 |
| 4 | 危险 API 扫描 | 关键词扫描与 AST 静态分析 | 需整改 | 发现 `shell=True` 与 `torch.load` 风险点。 |
| 5 | 依赖清单审查 | 检查 `requirements.txt` 固定版本 | 需复核 | 依赖版本较旧，建议联网环境使用 pip-audit 或 OSV 做 CVE 复核。 |

## 发现的问题

### SEC-001 [High] 防火墙命令通过 shell=True 执行

- 位置：`nids/nips.py:45`、`nids/nips.py:51`、`nids/nips.py:67`、`nids/nips.py:73`
- 证据：`subprocess.run(cmd, shell=True, check=True)`
- 影响：如果攻击者可影响 IP 输入，或 IP 值未经过严格校验，shell 元字符可能改变实际执行的防火墙命令。
- 建议：使用 `ipaddress.ip_address()` 校验 IP；将命令改为参数列表形式并设置 `shell=False`。

### SEC-002 [Medium] 模型文件通过 torch.load 加载

- 位置：`nids/model.py:78`、`nids/model.py:80`
- 证据：`torch.load(model_path...)`
- 影响：PyTorch 模型加载涉及反序列化；加载不可信模型文件可能带来不安全反序列化风险。
- 建议：仅加载可信模型文件；在支持的版本中使用 `weights_only=True`；对模型文件增加哈希或签名校验。

### SEC-003 [Low] 抓包逻辑需要高权限运行

- 位置：`nids/data_acquisition.py:35`
- 证据：`scapy.sniff(... filter="ip" ...)`
- 影响：抓包通常需要管理员或 root 权限，若应用整体高权限运行，会放大其他漏洞的影响。
- 建议：在部署说明中明确最小权限原则；尽可能拆分抓包、模型推理与防火墙操作权限。

## 依赖审查说明

`requirements.txt` 固定了如下依赖版本：

- `scapy==2.5.0`
- `numpy==1.26.0`
- `scipy==1.11.0`
- `torch==2.0.0`
- `torchvision==0.15.0`

上述依赖属于网络、科学计算和机器学习组件，版本相对较旧。建议在联网环境中使用 `pip-audit`、OSV Scanner 或 GitHub Dependabot 对当前漏洞公告进行复核，并根据复核结果升级到兼容的安全版本。

## GitHub 上传建议

建议将本材料包上传到仓库的 `security-test-evidence/` 目录，或作为申报材料附件单独上传。材料包内包含：

- `security-test-report.md`：中文安全测试报告
- `static-analysis-findings.json`：结构化扫描结果
- `evidence/file_inventory.txt`：源代码清单
- `evidence/dangerous_api_scan.txt`：危险 API 扫描证据
- `evidence/secret_keyword_scan.txt`：敏感信息扫描证据
- `evidence/requirements_review.txt`：依赖清单审查记录
