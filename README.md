<div align="center">

# 🐋 MoeCursor · DeepSeek 娘鼠标指针

**把 DeepSeek 娘做成你的鼠标指针！**

贴纸风格 · 蓝白配色 · 粉色小爱心 · 完整覆盖 Windows 11 全部 17 种指针

<img src="character_square.png" width="220" alt="DeepSeek 娘"/>

![预览](效果预览.png)

`Windows 11` `.cur 位图` `17 种指针角色` `32–256px 多尺寸`

</div>

---

## ✨ 这是什么

本仓库将 **DeepSeek 娘**（DeepSeek 的娘化吉祥物形象）制作成一套完整的 Windows 鼠标指针。
角色以白色贴纸 + 深蓝描边的形式出现在每个指针上：正常选择、文本选择的旁边她会探出头来，
"忙"的时候她占满整个指针，不可用时会帮她撑起一个红色禁止圈，后台运行时头顶还有一颗小星星。

![实际效果](真实效果预览.png)

## 📦 指针一览（17 个）

| 文件 | 角色 | 文件 | 角色 |
|---|---|---|---|
| moe_arrow.cur | 正常选择 | moe_hand.cur | 链接选择 |
| moe_help.cur | 帮助选择 | moe_busy.cur | 忙 |
| moe_appstart.cur | 后台运行 | moe_no.cur | 不可用 |
| moe_cross.cur | 精确选择 | moe_ibeam.cur | 文本选择 |
| moe_pen.cur | 手写 | moe_up.cur | 备用选择 |
| moe_sizens.cur | 上下调整 | moe_sizewe.cur | 左右调整 |
| moe_sizenwse.cur | 对角调整 1 | moe_sizenesw.cur | 对角调整 2 |
| moe_sizeall.cur | 移动 | moe_pin.cur | 位置选择 |
| moe_person.cur | 个人选择 | | |

每个 `.cur` 内置 **32 / 48 / 64 / 96 / 128 / 192 / 256px** 多档尺寸（32 位透明通道），
配合系统的指针大小设置自动切换，最大 256px 也不模糊。

## 🔧 安装（Windows 10 / 11）

1. 下载本仓库（`Code → Download ZIP`，解压后可得 `MoeCursor.zip`）
2. 右键 `安装鼠标指针.inf` → **安装**
3. `Win + R` 输入 `main.cpl` 回车 →「指针」选项卡 → 方案选 **MoeCursor** → 确定

调节大小：设置 → 辅助功能 → 鼠标指针和触控 →「大小」滑块。
恢复默认：同一界面方案选「Windows 默认（系统方案）」即可。

## 🎨 自己改

`build_v4.py` 是全部生成逻辑（Python + NumPy + OpenCV + Pillow），
改配色、浓度、爱心、角色大小等参数后重跑一遍就能重新生成整套指针。

## 📄 说明

- 指针文件与脚本按现状提供，随便用、随便改
- DeepSeek 娘角色形象版权归 DeepSeek 所有，本仓库仅将其制作成光标的形式分发，请勿用于商业用途
