# -*- coding: utf-8 -*-
"""Q4 report generator: paper-style Chinese research report (ReportLab route).
Consolidates tasks #1-#3 and P0b-P0h into docs/研究中期报告.pdf.
注意：正文内的引号一律用中文弯引号 “”，ASCII 双引号仅作 Python 定界符。
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "docs", "研究中期报告-流式Fiedler与谱变点检测.pdf")
ASSETS = os.path.join(ROOT, "docs", "report_assets")
os.makedirs(ASSETS, exist_ok=True)

REG = os.environ["DAIMON_CJK_FONT_REGULAR"]
BLD = os.environ["DAIMON_CJK_FONT_BOLD"]
pdfmetrics.registerFont(TTFont("CJK", REG))
pdfmetrics.registerFont(TTFont("CJK-Bold", BLD))
pdfmetrics.registerFontFamily("CJK", normal="CJK", bold="CJK-Bold",
                              italic="CJK", boldItalic="CJK-Bold")
for f in (REG, BLD):
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=REG).get_name()
plt.rcParams["axes.unicode_minus"] = False

NAVY = HexColor("#1a2e4a")
GRAY = HexColor("#666666")
BODY = HexColor("#222222")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="CJK-Bold",
                            fontSize=20, leading=28, textColor=NAVY, alignment=TA_CENTER),
    "subtitle": ParagraphStyle("st", parent=ss["Normal"], fontName="CJK",
                               fontSize=12, leading=18, textColor=GRAY, alignment=TA_CENTER),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName="CJK-Bold",
                         fontSize=15, leading=20, textColor=NAVY, spaceBefore=16,
                         spaceAfter=8),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="CJK-Bold",
                         fontSize=12.5, leading=17, textColor=NAVY, spaceBefore=10,
                         spaceAfter=5),
    "body": ParagraphStyle("b", parent=ss["Normal"], fontName="CJK", fontSize=10.5,
                           leading=16.5, textColor=BODY, alignment=TA_JUSTIFY,
                           spaceAfter=5),
    "abs": ParagraphStyle("ab", parent=ss["Normal"], fontName="CJK", fontSize=10,
                          leading=15.5, textColor=BODY, alignment=TA_JUSTIFY,
                          spaceAfter=4),
    "cap": ParagraphStyle("c", parent=ss["Normal"], fontName="CJK", fontSize=9,
                          leading=13, textColor=GRAY, alignment=TA_CENTER,
                          spaceBefore=4, spaceAfter=10),
    "ref": ParagraphStyle("r", parent=ss["Normal"], fontName="CJK", fontSize=9.5,
                          leading=14, leftIndent=22, firstLineIndent=-22,
                          textColor=BODY),
    "toc": ParagraphStyle("toc", parent=ss["Normal"], fontName="CJK", fontSize=10.5,
                          leading=20, textColor=BODY),
}


def fig_mdr_curves():
    recs = json.load(open(os.path.join(RUNS, "synth_mdr.json"), encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style = {0.0: ("o-", "#1a2e4a", "hub0（刚度 72）"),
             10.0: ("s-", "#8a3324", "hub10（刚度 797）"),
             30.0: ("^-", "#2e5e4e", "hub30（刚度 2396）")}
    for wh, (mk, col, lab) in style.items():
        th, at = [], []
        for d in sorted({r["delta"] for r in recs if r["w_hub"] == wh}):
            g = [r for r in recs if r["w_hub"] == wh and r["delta"] == d]
            th.append(sum(r["theta"] for r in g) / len(g))
            at.append(sum(r["att"] for r in g) / len(g))
        ax.plot(th, at, mk, color=col, label=lab, ms=4, lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("真旋转 θ（rad）")
    ax.set_ylabel(r"跟踪衰减 att = $d_{tr}/d_{ex}$")
    ax.axhline(1.0, color="#999999", lw=0.7, ls="--")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(frameon=False, fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    p = os.path.join(ASSETS, "fig_mdr_curves.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_mdr_law():
    recs = json.load(open(os.path.join(RUNS, "synth_mdr.json"), encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    pts = []
    for wh in [0.0, 10.0, 30.0]:
        for d in sorted({r["delta"] for r in recs if r["w_hub"] == wh}):
            g = [r for r in recs if r["w_hub"] == wh and r["delta"] == d]
            att = sum(r["att"] for r in g) / len(g)
            if att >= 0.5:
                pts.append((sum(r["c_minus_lam2"] for r in g) / len(g),
                            sum(r["theta"] for r in g) / len(g)))
                break
    xs = [p[0] ** 0.5 for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, "o", color="#1a2e4a", ms=7, label="受控实验 θ*")
    k = 1.2e-4
    xx = [min(xs) * 0.9, max(xs) * 1.1]
    ax.plot(xx, [k * x for x in xx], "-", color="#8a3324", lw=1.3,
            label=r"$\theta^{*} = 1.2\times 10^{-4}\cdot\sqrt{c-\lambda_{2}}$")
    for x, y in zip(xs, ys):
        ax.annotate("({:.0f}, {:.0e})".format(x, y), (x, y),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel(r"$\sqrt{c-\lambda_2}$（刚度平方根）")
    ax.set_ylabel(r"脱冻结阈值 $\theta^{*}$（rad）")
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    p = os.path.join(ASSETS, "fig_mdr_law.png")
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def three_line(data, widths, header_rows=1, font_size=9):
    cell = ParagraphStyle("cell", parent=S["body"], fontName="CJK",
                          fontSize=font_size, leading=font_size + 4,
                          alignment=0, spaceAfter=0)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="CJK-Bold")
    wrapped = [[Paragraph(str(c), cellb if r < header_rows else cell)
                for c in row] for r, row in enumerate(data)]
    t = Table(wrapped, colWidths=widths, repeatRows=header_rows)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "CJK"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 4),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, HexColor("#000000")),
        ("LINEBELOW", (0, 0), (-1, header_rows - 1), 0.6, HexColor("#000000")),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, HexColor("#000000")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def P(txt, style="body"):
    return Paragraph(txt, S[style])


def header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFont("CJK", 8.5)
    canvas.setFillColor(GRAY)
    canvas.drawCentredString(w / 2, h - 1.4 * cm,
                             "流式 Fiedler 向量维护与谱结构变点检测 · 研究中期报告")
    canvas.line(2.2 * cm, h - 1.6 * cm, w - 2.2 * cm, h - 1.6 * cm)
    canvas.drawCentredString(w / 2, 1.3 * cm, "— {} —".format(doc.page))
    canvas.restoreState()


def first_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(2)
    canvas.line(2.5 * cm, h - 3.2 * cm, w - 2.5 * cm, h - 3.2 * cm)
    canvas.line(2.5 * cm, 3.2 * cm, w - 2.5 * cm, 3.2 * cm)
    canvas.restoreState()


def main():
    story = []

    story.append(Spacer(1, 5.5 * cm))
    story.append(P("流式 Fiedler 向量维护与谱结构变点检测", "title"))
    story.append(Spacer(1, 0.4 * cm))
    story.append(P("—— 从跟踪器机制到检测可及性标度律 ——", "subtitle"))
    story.append(Spacer(1, 2.2 * cm))
    story.append(P("研究中期报告 v5（Q4 整编 + 查新闭环 + 校准依赖 + 断连 regime + G-REST 对比）", "subtitle"))
    story.append(P("项目：streaming-fiedler-research", "subtitle"))
    story.append(P("日期：2026 年 9 月 15 日（v1：9/14；v2：+LAD 双数据集；v3：+谱隙强扫描；"
                   "v4：+SCPD 精读/相关工作钉死、MDR 校准依赖、断连 regime 双证伪；"
                   "v5：+G-REST 全文精读/忠实复现/同台对比，detection-grade 定位确认）", "subtitle"))
    story.append(PageBreak())

    story.append(P('<a name="abstract"/>摘要', "h1"))
    story.append(P(
        "本报告整编“流式 Fiedler 向量维护 + 谱结构变点检测”研究线的全部实证工作（合成基线任务 #1–#3，"
        "真实数据标定 P0–P0b，向量漂移 P0c，Q1↔Q2 耦合标定 P0d，谱隙分桶 P0e，插桩裁决 P0f，"
        "修复验证 P0g，合成受控实验 P0h，LAD 基线对比 P0i，CollegeMsg 跨数据集 P0j，谱隙强扫描 P0k，"
        "查新闭环 P0l，tol 依赖 P0m，per-component 证伪 P0n，跟踪器轨迹证伪 P0o）。"
        "核心贡献有五：（1）在真实加权通信流（SNAP email-Eu）上发现并验证向量漂移与 λ<sub>2</sub> 平移是互补的变点信号，"
        "二者构成两通道检测器；（2）确立跟踪器对特征向量旋转的低通本质——早停容差将其锁为每样本单步"
        "投影器，跟踪位移与真旋转幅度统计解耦；（3）通过刚度（c−λ<sub>2</sub>≈2·d_max）与谱隙独立解耦的受控实验，"
        "仲裁真实流上观测到的负谱隙定律为刚度效应混淆，并标定最小可检测旋转的平方根刚度标度律 "
        "θ* ≈ 1.2×10<super>-4</super>·√(c−λ<sub>2</sub>) rad；进而用 4 块层级 SBM 图族把谱隙拉开 135 倍"
        "（0.13→17.5）而刚度近恒定，9/9 构型的 θ*/√(c−λ<sub>2</sub>) 全部落入 1.2×10<super>-4</super> 的约 2 倍常数带，"
        "再用 13 组校准组合扫描证明常数对精化容差 abs_tol 完全不敏感（4 个数量级零移动）、"
        "随重启门比 restart_tol_ratio 按幂律 α≈0.8 移动——MDR 律升级为 "
        "θ* ≈ 2.1×10<super>-4</super>·√(c−λ<sub>2</sub>)·(rtr/0.03)^0.8 的广义形态；"
        "（4）完整复现 KDD 2019 谱变点检测基线 LAD 并三通道同台：连通 regime（email-Eu）vec-D 8/8 严格占优"
        "（λ<sub>2</sub>-z 5/8、LAD 3/8），准断连 regime（CollegeMsg）vec-D 仍以 5/8 居首（LAD 2/8、λ<sub>2</sub>-z 退化 0/8），"
        "并揭示特征值通道家族在断连 regime 的数值退化机制；两条看似自然的修补路线"
        "（per-component 谱跟踪、流式热启动续传）均被实现并证伪，根源定位为结构性："
        "λ<sub>2</sub>≈0 时 Fiedler 向量是近零子空间的任意元，任何算法都无法稳定结构上任意的对象；"
        "（5）真实变点 day≈388 上呈现通道互补的另一侧证据："
        "LAD 与 vec-D 分列异常度排名前二，λ<sub>2</sub> 单标量近乎失明。全部实验资产随报告归档。", "abs"))
    story.append(P("<b>关键词</b>：Fiedler 向量；谱图论；流式算法；变点检测；代数连通度；随机游走拉普拉斯", "abs"))
    story.append(Spacer(1, 10))

    story.append(P("目录", "h1"))
    for name, anchor in [
        ("1　引言与问题定义", "sec1"), ("2　查新定位", "sec2"), ("3　方法", "sec3"),
        ("4　主要结果 I：真实流上的互补信号与真实变点", "sec4"),
        ("5　主要结果 II：Q1↔Q2 耦合与机制裁决", "sec5"),
        ("6　主要结果 III：受控实验、刚度仲裁、MDR 标度律、谱隙强扫描与校准依赖", "sec6"),
        ("7　讨论：设计平面耦合与自感知检测器", "sec7"),
        ("8　结论与展望", "sec8"), ("参考文献", "refs"),
        ("附录 A　实验资产清单", "appA")]:
        story.append(P('<a href="#{}" color="#1a2e4a">{}</a>'.format(anchor, name), "toc"))
    story.append(PageBreak())

    story.append(P('<a name="sec1"/>1　引言与问题定义', "h1"))
    story.append(P(
        "图谱的 Fiedler 值（λ<sub>2</sub>，代数连通度）与 Fiedler 向量（v<sub>2</sub>）是图结构最重要的谱量之一"
        '<super><a href="#ref1" color="black">[1]</a></super>，广泛用于谱聚类、图分割与网络结构分析'
        '<super><a href="#ref2" color="black">[2]</a></super>。本研究线关注其在<b>流式场景</b>下的两个耦合问题：', "body"))
    story.append(P(
        "Q1（跟踪）：边以流形式到达时，如何以远低于全量特征分解的成本维护 (λ<sub>2</sub>, v<sub>2</sub>)？"
        "Q2（检测）：λ<sub>2</sub>/v<sub>2</sub> 轨迹上的结构变点（community 重排、连接强度跃升）能否被可靠、"
        "低延迟地检出？Q3（耦合定理）：Q1 的近似误差如何决定 Q2 的可检测边界？", "body"))
    story.append(P(
        "先行合成基线（任务 #1–#3，见 5.1）给出两个负面但方向性的结论：动态 SBM 事件流上 λ<sub>2</sub> 无变点时"
        "即呈现约 23% 的随机游走式游荡，且增量噪声强度跨种子差 12 倍；以 MAD 归一为代表的中位数族"
        "检测器在重尾游荡期失效。由此确立主线：零模型选择、统计量族选择、以及真实数据标定。", "body"))

    story.append(P('<a name="sec2"/>2　查新定位', "h1"))
    story.append(P(
        "创新点查新（novelty-check 流程，2026-09-14）覆盖“流式/增量谱维护”与“谱变点检测”两组关键词；"
        "DOSSIER 全部候选文献已于当日经 arXiv/IEEE Xplore/Semantic Scholar/DBLP 多源二次核验"
        "（docs/查新核验-2026-09-14.md）：P1、P6、P14、P15、P18、P20、P21 等 CONFIRMED，"
        "P2 著录修正为 Lange 等 <i>Calc. Var. PDE</i> 2015（ frustration index + 磁 Cheeger 不等式）。"
        "核验的关键结论是发现<b>直接先例</b>：LAD（Laplacian Anomaly Detection，KDD 2019）"
        '<super><a href="#ref4" color="black">[4]</a></super>'
        "及其可扩展版 LADdos 已用拉普拉斯谱（特征值通道）+ 双滑窗做动态图变点检测，"
        "故“谱变点检测”本身不新；本工作的残差空间收窄并明确为三点组合：<b>边流式增量维护</b>"
        "（LAD 为每快照全量 SVD）、<b>特征向量漂移通道</b>（LAD 无此通道）、<b>三条规律级发现</b>"
        "（MDR 标度律、低通阶最优、通道互补）。反向检索（针对三个新结果）未见等价工作。", "body"))
    story.append(P(
        "<b>SCPD 精读闭环（P0l）</b>：SCPD 著录钉死为 Huang 等 <i>PAKDD 2023</i>"
        '<super><a href="#ref5" color="black">[5]</a></super>（'
        "arXiv:2305.08750），全文精读确认其属性通道为 LDOS——属性向量与全谱特征向量对齐度的"
        "分桶分布跟踪，仅用于属性变点；本工作的 vec-D 通道（无属性、钉住 v<sub>2</sub> 单一特征对、"
        "由跟踪器逐样本产出的角漂移）在语义、对象、用途上三点不同，<b>通道先例状态 CONFIRMED 保住</b>。"
        "中文核心（计算机学报/软件学报）与数学侧（MathSciNet 范围：Fiedler 摄动导数、去中心化 λ<sub>2</sub> "
        "估计、鲁棒 Fiedler 估计）补检索均无同题命中——三条规律级发现的反向检索依然干净。", "body"))
    story.append(P(
        "<b>新增相关工作义务（P0l）</b>：Q1 跟踪器赛道存在直接先例——G-REST"
        '<super><a href="#ref6" color="black">[6]</a></super>（'
        "Eini 等，arXiv 2026-03：Rayleigh-Ritz 投影子空间跟踪演化图的邻接与拉普拉斯主导特征对）、"
        "TRIP" '<super><a href="#ref7" color="black">[7]</a></super>（'
        "Chen &amp; Tong，SDM 2015：一阶摄动特征函数跟踪）、RSDV"
        '<super><a href="#ref8" color="black">[8]</a></super>（'
        "Kalantzis &amp; Traganitis，ICASSP 2023：预解展开特征嵌入更新）。"
        "因此“流式特征对维护”单独不构成贡献点；本工作的定位相应明确为<b>跟踪-检测耦合与检测可及性"
        "标度律</b>：跟踪器是带插桩的诊断装置而非又一个维护算法，核心增量是三条规律级发现、"
        "vec-D 互补通道与断连 regime 机制研究——这些是上述三篇所无。G-REST 目前仅挂 arXiv，"
        "在其正式发表前投稿可锁定优先权。", "body"))
    story.append(P(
        "<b>G-REST 精读与同台对比闭环（P0p）</b>：G-REST 全文（21 页）精读并忠实复现其 Alg. 2"
        "（Rayleigh-Ritz 投影子空间跟踪；固定节点集时子空间退化为 span{X_K, (I−X_K X_Kᵀ)ΔX_K}；"
        "拉普拉斯端须经移位 T=cI−L 到达——直接跟踪 −L 会被投影子空间多出的方向在谱边缘 0 处产生的"
        "虚假 Ritz 值污染 top-K 选择，此为其框架的正确性前提而论文未明说）。在 P0h 受控装置上以"
        "流传模式同台对比：G-REST 单步精度很好（δ=1e-3 时 λ<sub>2</sub> 误差 1e-9），"
        "但<b>无重启/无验证机制导致逐步累积</b>——逐步转角捕获率仅 1–6.5% 且随 δ 漂移，"
        "终态特征向量漂移 3e-4–8e-4、λ<sub>2</sub> 误差 0.03–0.06；"
        "本工作的 warmstart 跟踪器在 follow-or-restart 策略下终态恒回机器精度（每 11 步仅重启 1–3 次）。"
        "G-REST 单步快约 3.5 倍，故本工作卖点不是更快，而是<b>detection-grade 保真度</b>："
        "对变点检测而言，跟踪输出中的转角本身就是待测信号，G-REST 相当于对其施加未知、状态依赖、"
        "1–6% 的低通衰减，检测阈值在其输出上不可校准；其缺的“何时重启/如何验证”正是本工作"
        "带校准的重启策略与检测可及性律（θ*≈k·√(c−λ<sub>2</sub>)）所回答的问题。", "body"))

    story.append(P('<a name="sec3"/>3　方法', "h1"))
    story.append(P("3.1　跟踪器：WarmStartTracker（含插桩与修复）", "h2"))
    story.append(P(
        "跟踪器保留上一状态的 (λ<sub>2</sub>, v<sub>2</sub>, λ<sub>3</sub>)。每样本以旧 v<sub>2</sub> 的 Rayleigh 商与残差做门控："
        "残差 ≤ restart_tol×(λ<sub>3</sub>−λ<sub>2</sub>) 时做幂迭代精化（从旧向量热启动、每步对平凡特征向量做收缩），"
        "否则精确重启（shift-invert ARPACK）。P0f 插桩发现 gap 相对早停容差（0.03×gap）把精化步数"
        "锁死为 k=1（84/84 样本），使跟踪器成为“每样本单步投影器”；P0g 引入绝对容差 refine_abs_tol=1e-6，"
        "步数变为自适应（med 1 / max 30），CP 处跟踪信号提升 28–326 倍，成本仅 0.5→1.3 ms/样本"
        "（精确解约 75 ms）。插桩同时逐样本记录分支、残差、gap 估计与 d_max。", "body"))
    story.append(P("3.2　两通道检测器", "h2"))
    story.append(P(
        "通道 A（标度/慢漂移）：λ<sub>2</sub> 轨迹的窗口 z（基线 30 / 近期 10，局部 MAD 归一）。"
        "通道 B（形状/拓扑重排）：v<sub>2</sub> 余弦漂移 D=1−⟨v_t, v_{t−10}⟩ 的窗口 z，以及逐点漂移 "
        "d=1−⟨v_t, v_{t−1}⟩ 的原始幅度尖峰检测。两通道阈值均按估计族在控制段分别校准"
        "（P0d/P0g 教训），并加 ×(1+1e<super>-6</super>) 相对裕量以消除跨进程特征求解的位级不确定性"
        "导致的 1-ulp 假报警。", "body"))
    story.append(P("3.3　实验装置", "h2"))
    story.append(P(
        "真实流：SNAP email-Eu 时间戳邮件网络"
        '<super><a href="#ref3" color="black">[3]</a></super>，'
        "限制在 top-300 活跃核心（否则 λ<sub>2</sub> 恒为零），指数衰减记忆（τ≈20k 边）下每 500 边采样一次，"
        "共 466 样本，跨度约 800 天；注入型变点（批量删边 / 全局缩放）在全局样本 400 处触发。"
        "合成受控：两块 SBM + 可插拔重枢纽（独立抬升刚度 c−λ<sub>2</sub>=2·d_max+1−λ<sub>2</sub>）+ 重定位/细权边"
        "两种可控旋转旋钮。", "body"))

    story.append(P('<a name="sec4"/>4　主要结果 I：真实流上的互补信号与真实变点', "h1"))
    story.append(P("4.1　真实流标定（P0/P0b）：修正两个前提", "h2"))
    story.append(P(
        "真实 email-Eu 流 λ<sub>2</sub> 噪声远低于合成基线：稳态段相对游荡仅 4.4%（合成 23%），"
        "即合成的随机游走噪声主要是模型 artifact。但“OU 型均值回归”的前提需修正：AR(1) 拟合 "
        "φ=0.999（166 样本无法拒绝单位根），该段实为 +105% 的持续爬坡（day 389→519），"
        "真实 λ<sub>2</sub> 是“趋势 + 小创新量”而非平稳 OU。检测器对比（控制组零误报预算校准）："
        "窗口 z 对删除型注入 CP 3/6、对单调缩放型 2/2；OU 残余 CUSUM 0/8——其阈值由慢漂移"
        "累计校准（约 60σ），对单点跳变结构性失明，但放松预算后首位报警恰好落在真实变点处。", "body"))
    story.append(P(
        "<b>真实变点修正</b>：P0 原报 day≈510 候选变点经创新量分析修正为 <b>day≈388</b>——"
        "全段最大单点创新量 8.46σ 与残余 CUSUM 放松预算首位报警均落在 day 388（爬坡起点），"
        "day 510 只是爬坡高点。另发现真实图特有现象：批量删边引起的 λ<sub>2</sub> 平移<b>符号随机</b>"
        "（删陈旧重边反而抬高 λ<sub>2</sub>），故“删边比例”不是可控的 CP 强度参数。", "body"))
    story.append(P("4.2　向量漂移与 λ<sub>2</sub> 互补（P0c）", "h2"))
    story.append(P(
        "累积向量漂移 D 的窗口 z 在 8 条注入轨迹上 <b>8/8 检出、零误报</b>，对照 λ<sub>2</sub>-z 的 5/8——"
        "补获的 3 条恰好全是 λ<sub>2</sub> 漏检的“删边反抬 λ<sub>2</sub>”正向平移案例。机理上两通道感知 CP 的不同侧面："
        "均匀缩放边权对 v<sub>2</sub> 方向严格不变（vec-D 延迟 26 样本），而 λ<sub>2</sub> 即时响应（延迟 11）。"
        "逐点漂移原始幅度在 day 388 真实变点尖峰 2800 倍于中位数，并对 3 条 λ<sub>2</sub> 漏检 run 在 CP+0 "
        "零延迟全部检出；但中位数基线窗口 z 对单点尖峰结构性失明（尖峰同时进入近期窗与基线窗，"
        "两个中位数都不动）——尖峰通道必须走原始幅度/max 型统计量。", "body"))
    data = [["通道（8 条注入 CP，fp=0 校准）", "检出", "备注"],
            ["vec-D 窗口 z（精确 v<sub>2</sub>）", "8/8", "延迟 5–26 样本；补获 3 条 λ<sub>2</sub> 盲案例"],
            ["λ<sub>2</sub> 窗口 z", "5/8", "漏检均为正向平移（删边反抬 λ<sub>2</sub>）"],
            ["逐点 d 原始尖峰", "3/3（λ<sub>2</sub> 漏检集）", "零延迟；day 388 尖峰 2800×中位"],
            ["OU 残余 CUSUM", "0/8", "对单点跳变失明；放松预算命中 day 388"]]
    story.append(KeepTogether([P('<a name="tab1"/>表 1　真实流注入 CP 的四统计量对比', "cap"),
                               three_line(data, [6.2 * cm, 2.6 * cm, 7.6 * cm])]))

    story.append(P("4.3　与 KDD 2019 基线 LAD 的双数据集对比（P0i/P0j）", "h2"))
    story.append(P(
        "将 LAD 完整复现（拉普拉斯 top-6 奇异值签名 + 双滑窗 5/10 + 主成分正常行为 + 余弦距离 "
        "Z + 增量分 Z*），与 λ<sub>2</sub>-z、vec-D 在<b>同一批重放流</b>上同台（重放保真 max_rel_diff≈1e-9，"
        "有效段 Fiedler 向量 |cos|=1.000000），三通道统一按控制段零误报预算 + 1e-6 余量校准。"
        "两个数据集覆盖两种 regime：email-Eu 全程连通（段内 λ<sub>2</sub>∈[6.3,13]、谱隙≥40），"
        "CollegeMsg 准断连（段内 98% 样本 λ<sub>2</sub>&lt;0.05，连通率仅 1.9%）。", "body"))
    data = [["数据集（regime）", "vec-D", "λ<sub>2</sub>-z", "LAD", "关键现象"],
            ["email-Eu（连通）", "<b>8/8</b>", "5/8", "3/8",
             "LAD 检出是 vec-D 子集；延迟 vec-D 5–26 vs LAD 5–22"],
            ["CollegeMsg（准断连）", "<b>5/8</b>", "0/8", "2/8",
             "λ<sub>2</sub>-z 数值退化（幻影 fp=5）；vec-D 单 run 10 fp"]]
    story.append(KeepTogether([P('<a name="tab2"/>表 2　双 regime 三通道检出（8 注入 run，fp 见正文）', "cap"),
                               three_line(data, [4.0 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 7.3 * cm])]))
    story.append(P(
        "<b>LAD 的盲区结构</b>：（1）全局缩放 CP（sc06）下签名向量整体缩放、余弦距离尺度不变，"
        "email-Eu 上完全失明（λ<sub>2</sub>-z 反而 2/2 检出）；（2）三个删边型漏检 run 与 λ<sub>2</sub>-z 的漏检集合"
        "<b>完全重合</b>——LAD 与 λ<sub>2</sub>-z 同属特征值通道家族，共享“删边反抬 λ<sub>2</sub>”类案例的通道局限；"
        "vec-D 的补获恰好全在这两个集合里。（3）sc06 失明是 regime 依赖的：CollegeMsg 上 LAD 检出 "
        "sc06b（延迟 3）——脆弱图上缩放后新边混入立即改变谱射线方向，检测来自后继演化而非缩放本身。", "body"))
    story.append(P(
        "<b>真实变点 day≈388 的通道排名</b>（控制流全序列异常度）：LAD 列 <b>第 0/160</b>，"
        "vec-D 列 <b>第 1/156</b>，λ<sub>2</sub>-z 仅列第 127/166——大结构变化上特征值通道（含 LAD）最灵敏、"
        "λ<sub>2</sub> 单标量近乎失明；微小删边变化上 vec-D 独自全覆盖。<b>“特征值通道与特征向量通道互补”"
        "在注入集与真实事件两侧均获证据。</b>", "body"))
    story.append(P(
        "<b>断连 regime 的数值退化（新发现）</b>：λ<sub>2</sub>≈0 时增量 MAD 打到 1e-12 地板，"
        "特征求解器跨 run 的 1e-18 级噪声被放大为 z~1e-7 并越过由同类噪声设定的校准阈值，"
        "λ<sub>2</sub>-z 出现幻影误报——该通道在稀疏断连流上<b>作为设计不成立</b>。vec-D 同样受零空间求解器"
        "噪声穿透（单 run 10 个误报），但程度轻一个数量级。试修的“连通性门控”变体（全窗连通才评估）"
        "在 1.9% 连通率下近乎全弃权，不可行。", "body"))
    story.append(P(
        "<b>两条修补路线的实现与证伪（P0n/P0o）</b>：针对该 regime 的两条看似自然的展望——"
        "per-component 谱跟踪与流式热启动续传——均已实现并证伪，且失败机制各不相同："
        "（1）主导分量跟踪：分量身份抖动（主导分量节点集在样本间 234↔500 波动，ncomp 在 1–440 间"
        "碎裂-重组）使 vec-D^dom 在控制流上中位 D=0.957（无法校准），λ<sub>2</sub>^dom 被自然增密趋势淹没"
        "（控制流自身 0.95→1.96，注入同幅度），ncomp 既无 CP 印迹（删边不改变分量计数）又中 sig 地板病理；"
        "（2）热启动续传：给跟踪器加绝对门地板（gate_floor/verify_floor，默认保持旧行为）使精化分支"
        "持续工作，重启率降至 12–21%，但精化后残差中位 0.5–0.6——续传向量约一半能量在 near-zero "
        "特征空间外，产出的是由真实图 churn 驱动的随机游走，控制流 vec-D 中位 0.803，比逐快照 eigsh 更差。"
        "<b>合取结论：困难根源是结构性而非数值性</b>——λ<sub>2</sub>≈0 且重数随碎裂 churn 时，"
        "“该快照的 Fiedler 向量”是近零子空间的任意元，eigsh 给出一个任意元，确定性续传给另一个"
        "不收敛的游走，任何算法都无法稳定结构上任意的对象。该 regime 需要的是不同的 CP 语义"
        "（如 per-component 分层 SBM regime 图），而非同一语义的谱通道换底。正面信号：沿轨迹的 "
        "Rayleigh 商（lam-trk）以 2/8 优于逐快照 λ<sub>2</sub>-z 的 0/8——轨迹 Rayleigh 商是更平滑的"
        "特征值通道统计量（7 fp 尚不可用，作观察记录）。", "body"))

    story.append(P('<a name="sec5"/>5　主要结果 II：Q1↔Q2 耦合与机制裁决', "h1"))
    story.append(P("5.1　合成基线（任务 #1–#3）", "h2"))
    data = [["任务", "设置", "结果"],
            ["#1 漂移本底", "动态 SBM 事件流，无 CP", "λ<sub>2</sub> 相对游荡约 23%；σ_inc 跨 seed 差 12 倍 → 检测器须局部自适应归一"],
            ["#2 检测器 v2", "MAD 归一窗口 z vs 朴素 CUSUM", "负面结果：MAD 在重尾游荡期失效，v2 不敌朴素 CUSUM；临界 SNR≈1.5"],
            ["#3 前沿 scaling", "多统计量阈值校准", "深层负面：零分布重尾使阈值不可传递，无干净 SNR 前沿 → 主线转向换统计量/真实数据标定"]]
    story.append(KeepTogether([P('<a name="tab3"/>表 3　合成基线三任务摘要', "cap"),
                               three_line(data, [2.6 * cm, 4.6 * cm, 9.2 * cm])]))
    story.append(P("5.2　耦合标定（P0d）：失效突发相与低通本质", "h2"))
    story.append(P(
        "精确解与跟踪解双解重放显示：默认参数下跟踪器在稀疏低谱隙起始段（day 357–380）连续 22 样本"
        "收敛到错误特征方向（v̂ 与真值正交、λ̂ 偏高 8–17 倍），向量漂移出现 0.99 巨型伪峰——"
        "<b>Q1 失效直接转化为 Q2 假变点</b>。restart_tol 0.1→0.03 零成本消除突发（重启数与耗时不变）。"
        "可靠相上跟踪质量优秀（λ<sub>2</sub> 相对误差中位 1.8e-4，v 误差中位 2.0e-3）。但注入 CP 处真旋转 "
        "d_ex=4.8e-4 经跟踪后只剩 d_tr≈1.3e-7——<b>衰减 3.5 个数量级</b>；按估计族分别校准阈值后"
        "raw-spike 4/4、vec-D-z 3/4 检出，SNR 代价从 1e5 降至 29。即检测可及性由跟踪器收敛速度"
        "而非统计量决定。", "body"))
    story.append(P("5.3　机制裁决（P0e/P0f）：k=1 与符号悖论", "h2"))
    story.append(P(
        "谱隙分桶显示衰减对谱隙稳健负相关（Spearman −0.507，时间分层下依然成立），与朴素幂迭代"
        "收缩论证的符号相反。插桩裁决：84 对全部走精化接受分支（门控假说否决）、<b>精化步数恒为 1</b>"
        "（gap 相对早停一步即达标）、朴素单步几何模型否决（log-log Pearson −0.31，预言平坦 1.3e-3 "
        "对实测 8.3e-2→0）。最硬的事实：<b>log d_tr 对 log d_ex 斜率 −0.04</b>——跟踪位移与真旋转"
        "幅度完全不相关；d_tr 由跟踪器自身归一化残差 res_in/(c−λ<sub>2</sub>) 驱动（Spearman +0.64）。"
        "跟踪器是自指的：只对自身残差几何敏感，对图的真旋转失明。", "body"))
    story.append(P("5.4　修复验证（P0g）", "h2"))
    story.append(P(
        "绝对容差 1e-6 解除单步锁死：k 变为 med 1 / max 30 自适应，CP 信号 ×28–326，跟踪误差 ↓1.5×，"
        "成本 0.5→1.3 ms/样本。但本底同步变重尾（max 6.3e-8→5.7e-6）——k=1 锁死原是偶然噪声滤波器。"
        "q99 分位校准下两版均 3/3 检出且延迟改善；严格 max 零虚警校准下修复版退为 1/3。<b>结论：跟踪容差"
        "与检测校准是设计平面上的耦合，必须配对选择。</b>", "body"))
    data = [["量", "修复前（gap 相对容差）", "修复后（abs 1e-6）"],
            ["精化步数 k", "恒 1", "med 1 / max 30"],
            ["CP 处漂移信号", "基线", "×28–326"],
            ["跟踪本底 max", "6.3e-8（紧）", "5.7e-6（重尾）"],
            ["q99 校准检出", "3/3（延迟 22/0/0）", "3/3（延迟 19/0/0）"],
            ["max 校准检出", "3/3", "1/3"]]
    story.append(KeepTogether([P('<a name="tab4"/>表 4　绝对容差修复前后对比', "cap"),
                               three_line(data, [4.2 * cm, 6.0 * cm, 6.2 * cm])]))

    story.append(P('<a name="sec6"/>6　主要结果 III：受控实验、刚度仲裁、MDR 标度律、谱隙强扫描与校准依赖', "h1"))
    story.append(P(
        "P0e 的负谱隙定律在真实流上无法与刚度效应分离（gap 与 d_max 沿 densification 爬坡共变）。"
        "受控实验将二者解耦：重枢纽使刚度 c−λ<sub>2</sub> 跨 33 倍（72→2396）而谱隙仅变 20%（5.95–7.44）。"
        "结果分三区：大旋转（θ≥0.12 rad）att=1.000（252 配置全部，刚度无关）；小旋转平台区 att 被"
        "刚度钉死——hub0 ≈1.0、hub10 ≈0.20、hub30 ≈0.03，与 θ 无关；过渡区给出脱冻结阈值 "
        "θ*（72→1e-3、797→3e-3、2396→1e-2 rad）。<b>负谱隙定律仲裁为刚度效应混淆。</b>", "body"))
    story.append(Image(fig_mdr_curves(), width=14.6 * cm, height=7.3 * cm))
    story.append(P('<a name="fig1"/>图 1　跟踪衰减 att 对真旋转 θ：三档刚度的低通平台与过渡区', "cap"))
    story.append(P(
        "三档阈值满足平方根刚度律：<b>θ* ≈ 1.2×10<super>-4</super>·√(c−λ<sub>2</sub>) rad</b>（检验：72→1.0e-3、797→3.4e-3 "
        "对实测 3e-3、2396→5.9e-3 对实测 1e-2）。配上各档 SNR≥1 的实测最小可检测旋转（1e-3 / 3e-3 / "
        "1e-2 rad），跟踪精度与检测灵敏度的耦合获得闭式形式：可检测旋转 ≥ 1.2×10<super>-4</super>·√(2·d_max−λ<sub>2</sub>)。"
        "这是 Q3 耦合定理的经验-解析链条终点：经验定律（P0e）→ 机制（P0f 单步锁死 / P0g 修复）→ "
        "受控仲裁（P0h）。", "body"))
    story.append(Image(fig_mdr_law(), width=12.6 * cm, height=6.7 * cm))
    story.append(P('<a name="fig2"/>图 2　脱冻结阈值 θ* 对 √(c−λ<sub>2</sub>)：平方根刚度律拟合', "cap"))
    data = [["刚度档 c−λ<sub>2</sub>", "att 平台值", "θ*（rad）", "MDR（SNR≥1）"],
            ["72（hub0）", "≈1.0", "≤1e-6", "θ ≈ 1e-3"],
            ["797（hub10）", "≈0.20", "≈3e-3", "θ ≈ 3e-3"],
            ["2396（hub30）", "≈0.03", "≈1e-2", "θ ≈ 1e-2"]]
    story.append(KeepTogether([P('<a name="tab5"/>表 5　受控实验：刚度平台与 MDR', "cap"),
                               three_line(data, [4.4 * cm, 3.2 * cm, 3.4 * cm, 5.4 * cm])]))

    story.append(P("6.1　谱隙强扫描：常数带验证（P0k）", "h2"))
    story.append(P(
        "P0h 中 gap 仅跨 20%，“θ* 由刚度而非谱隙控制”仅为弱支持。为此设计 4 块层级 SBM 图族："
        "远耦合 {1,2}|{3,4} 钉死 λ<sub>2</sub>，近耦合按边条数扫描（p_near 60×）实现 gap 大扫而 λ<sub>2</sub> 与 "
        "d_max 仅轻度变动；枢纽 dial 保持刚度近恒定（hub10/30 时 c−λ<sub>2</sub> 在各组间变化 &lt;0.5%）。"
        "谱隙实扫 <b>0.13→17.5（135×）</b>。θ* 取过渡区间 [末个 att&lt;0.5, 首个 att≥0.5] 的几何中点"
        "（网格分辨率 ~3× 为诚实不确定度），结果：", "body"))
    data = [["谱隙 gap", "刚度 c−λ<sub>2</sub>", "θ* 区间（rad）", "k=θ*/√(c−λ<sub>2</sub>)"],
            ["0.13 – 0.31", "45 – 2399", "[6e-4, 9.7e-3]", "1.57e-4 / 7.8e-5 / 1.14e-4"],
            ["1.76 – 1.88", "799 / 2399", "[2.0e-3, 6.8e-3]", "1.26e-4 / 8.1e-5"],
            ["4.65 – 5.20", "799 / 2399", "[4.0e-3, 1.3e-2]", "2.43e-4 / 1.52e-4"],
            ["15.8 – 16.8", "799 / 2398", "[1.7e-3, 1.5e-2]", "1.05e-4 / 1.83e-4"]]
    story.append(KeepTogether([P('<a name="tab6"/>表 6　谱隙强扫描：9/9 构型的常数带', "cap"),
                               three_line(data, [3.4 * cm, 3.4 * cm, 4.4 * cm, 5.2 * cm])]))
    story.append(P(
        "<b>9/9 构型 k ∈ [7.8e-5, 2.4e-4]，全部落入 P0h 常数 1.2×10<super>-4</super> 的约 2 倍带内</b>。"
        "固定刚度层内 70× 的谱隙变化只移动 θ* 约 2.5 倍（log-log 斜率 0.13–0.14、R²≤0.43，"
        "≈网格分辨率），合并双因子回归 log θ* ~ a·log(c−λ<sub>2</sub>) + b·log(gap) 给出 a=+0.37、b=+0.12——"
        "刚度主控、谱隙至多弱正残差。附带发现：低刚度不冻结区（hub0、gap≥1.9，θ*&lt;1e-6）在新图族"
        "独立复现；gap→0 近简并重新打开冻结，但属特征空间混叠机制（v<sub>2</sub> 病态），不并入 MDR 律。"
        "层级图族本身成为“λ<sub>2</sub> 钉死、gap 大扫”约束下的可复用标准装置。", "body"))

    story.append(P("6.2　校准旋钮依赖：abs_tol 不变性与重启门幂律（P0m）", "h2"))
    story.append(P(
        "常数带是否在跟踪器自身的校准选择下幸存？在 P0k 标准装置的 8 个常数带内构型上扫描 13 组校准组合"
        "（refine_abs_tol ∈ {1e-4, 1e-6, 1e-8, 1e-10} × restart_tol_ratio ∈ {0.01, 0.03, 0.1} + 遗留 "
        "gap 相对模式），δ 网格加密至 33 点（θ* 分辨率 ~1.15×）：", "body"))
    data = [["校准组合（abs_tol, rtr）", "k 中位", "相对参考", "成本（精化步/更新）"],
            ["(1e-4 – 1e-10, 0.01)", "8.5e-5", "0.41×", "13–19"],
            ["(1e-4 – 1e-10, 0.03) 参考", "2.06e-4", "1.00×", "16–22"],
            ["(1e-4 – 1e-10, 0.1)", "5.25e-4", "2.55×", "19–25"],
            ["(None, 0.03) gap 相对（k=1 锁死）", "2.35e-4", "1.14×", "0.7"]]
    story.append(KeepTogether([P('<a name="tab7"/>表 7　校准组合扫描：k 的 tol 依赖（8 构型 × 5 seeds）', "cap"),
                               three_line(data, [6.4 * cm, 2.6 * cm, 2.6 * cm, 4.8 * cm])]))
    story.append(P(
        "<b>两个结论</b>：（1）<b>k 对 refine_abs_tol 完全不敏感</b>——四个数量级的容差档位 att 曲线"
        "逐点相同，解冻过渡由重启门（res ≤ rtr·gap_est）控制，abs_tol 只决定精化循环内部何时停，"
        "MDR 常数不是精化阈值的偶然产物；（2）<b>k 随 restart_tol_ratio 按幂律 α≈0.79 移动</b>"
        "（10× 旋钮 → 6.2× k，超出 1.41× 网格分辨率，真实效应）——门越松，低通平台吸收越多旋转，"
        "att&lt;0.5 维持到更大 θ。<b>广义 MDR 律：θ* ≈ 2.1×10<super>-4</super>·√(c−λ<sub>2</sub>)·(rtr/0.03)^0.8</b>，"
        "对 abs_tol ∈ [1e-10, 1e-4] 不变。附带：P0f 的 k=1 锁死在第二图族独立复现且解冻阈值同带"
        "（1.14×）——锁死改变平台行为、不移解冻阈值；θ*（检测可及性）与精化步数（运行成本）跨旋钮"
        "近似解耦，定量支撑第 7 节的设计平面耦合论点。", "body"))

    story.append(P('<a name="sec7"/>7　讨论：设计平面耦合与自感知检测器', "h1"))
    story.append(P(
        "本工作链最重要的方法论结论是：<b>跟踪器与检测器必须在设计平面上共同设计</b>。"
        "跟踪容差（gap 相对 vs 绝对）、精化步数上限、重启门限与检测阈值校准规则（max vs 分位数）"
        "构成一个联合选择，单独优化任一侧都会给出误导性结论（P0g 的严格 max 校准陷阱）。"
        "由此形成“自感知检测器”的具体形态：λ<sub>2</sub>-z 通道（慢漂移/标度 CP）+ vec-D 通道（拓扑重排）"
        "+ 逐点尖峰通道（零延迟大旋转）+ 跟踪健康度分相（低谱隙稀疏段只信精确重启）。", "body"))
    story.append(P("诚实边界", "h2"))
    story.append(P(
        "（1）单步注入协议 ≠ 流式累积，连续小旋转的检出由 vec-D 累积机制覆盖（P0c）；"
        "（2）LAD 复现采用作者推荐窗长（5/10）未调参，存在调参空间，但零误报校准协议对三通道一致公平；"
        "（3）谱隙强扫描中 gap→0 近简并构型（v<sub>2</sub> 病态）的机制未解析化，仅标注排除；"
        "（4）MDR 律的校准依赖已部分解决（P0m）：abs_tol 不变性成立（4 个数量级），"
        "rtr 幂律仅在 0.01–0.1 范围内表征，更宽范围的形态未知；"
        "（5）查新核验为书目级（未精读付费墙全文），P6“未做 Cheeger 界”基于摘要+目录级浏览；"
        "SCPD 已精读（通道先例保住），G-REST 已完成全文精读、忠实复现与同台对比（P0p，"
        "结论：embedding-grade 跟踪不满足 detection-grade 保真度要求）；TRIP/RSDV 为 G-REST "
        "论文内已对比的一阶/预解基线，直接引用其报告数值即可；"
        "（6）CollegeMsg 准断连 regime：vec-D 的 5/8 是在通道语义部分断裂（结构性）条件下取得的，"
        "其 inj050b 的 10 个误报经 P0n/P0o 证伪两条修补路线后标注为结构性不可修；"
        "断连 regime 的全部结论以“per-snapshot 谱通道”为条件。", "body"))

    story.append(P('<a name="sec8"/>8　结论与展望', "h1"))
    story.append(P(
        "结论：流式 Fiedler 维护与谱变点检测之间存在可定量刻画的耦合。在真实加权通信流上，"
        "λ<sub>2</sub> 平移与 v<sub>2</sub> 漂移构成互补检测通道，且该互补对 KDD 2019 基线 LAD 严格占优"
        "（连通 regime 8/8 vs 3/8）并跨 regime 稳健（准断连 regime 5/8 vs 2/8）；热启动跟踪器对特征向量"
        "旋转是单步投影器式低通，其可检测边界由刚度（2·d_max−λ<sub>2</sub>）以平方根律控制而非谱隙；"
        "跟踪容差与检测校准必须联合设计；断连 regime 下谱通道的失效根源是结构性"
        "（Fiedler 向量在 near-zero 子空间内任意），而非可修的数值噪声。"
        "展望：（1）断连 regime 的新 CP 语义设计（per-component 分层 SBM regime 图、"
        "组件群落级事件），替代已证伪的通道换底路线；（2）把平方根刚度律解析化"
        "（单步投影 + 早停的完整推导，含 gap→0 近简并机制）；（3）投稿路线：主目标 "
        "ECML-PKDD 2027（截稿约 2027-03，会期 2027-09，Eindhoven）——与 LAD/SCPD 谱系"
        "同社区；KDD 2027（截稿约 2027-02）视理论化进展冲刺；ICDM 2027 兜底。"
        "G-REST 目前仅挂 arXiv（2026-03），应尽快投稿锁定优先权。", "body"))

    story.append(P('<a name="refs"/>参考文献', "h1"))
    story.append(Paragraph('<a name="ref1"/>[1] Fiedler M. Algebraic connectivity of graphs. '
                           '<i>Czechoslovak Mathematical Journal</i>, 1973, 23(98): 298-305.', S["ref"]))
    story.append(Paragraph('<a name="ref2"/>[2] von Luxburg U. A tutorial on spectral clustering. '
                           '<i>Statistics and Computing</i>, 2007, 17(4): 395-416.', S["ref"]))
    story.append(Paragraph('<a name="ref3"/>[3] Leskovec J, Kleinberg J, Faloutsos C. Graph evolution: '
                           'Densification and shrinking diameters. <i>ACM Transactions on Knowledge '
                           'Discovery from Data</i>, 2007, 1(1): 2-es.（email-Eu 数据集来源，SNAP）', S["ref"]))
    story.append(Paragraph('<a name="ref4"/>[4] Huang S, Chawla S, Eaton P, et al. Laplacian anomaly '
                           'detection for dynamic graphs（LAD）. <i>Proceedings of the 25th ACM SIGKDD '
                           'International Conference on Knowledge Discovery and Data Mining</i>, 2019: '
                           '2238-2246.', S["ref"]))
    story.append(Paragraph('<a name="ref5"/>[5] Huang S, Danovitch J, Rabusseau G, Rabbany R. '
                           'Fast and attributed change detection on dynamic graphs with density of '
                           'states（SCPD）. <i>Pacific-Asia Conference on Knowledge Discovery and '
                           'Data Mining（PAKDD）</i>, 2023: 15-26.', S["ref"]))
    story.append(Paragraph('<a name="ref6"/>[6] Eini M, Karaaslanli A, Kalantzis V, Traganitis P A. '
                           'Subspace projection methods for fast spectral embeddings of evolving '
                           'graphs（G-REST）. arXiv:2603.19439, 2026.', S["ref"]))
    story.append(Paragraph('<a name="ref7"/>[7] Chen C, Tong H. Fast eigen-functions tracking on dynamic graphs. '
                           '<i>Proceedings of the 2015 SIAM International Conference on Data '
                           'Mining（SDM）</i>, 2015: 559-567.', S["ref"]))
    story.append(Paragraph('<a name="ref8"/>[8] Kalantzis V, Traganitis P A. Matrix resolvent eigenembeddings '
                           'for dynamic graphs. <i>IEEE International Conference on Acoustics, '
                           'Speech and Signal Processing（ICASSP）</i>, 2023.', S["ref"]))
    story.append(Paragraph('[9] Hu S, 等. LADdos：基于网络态密度近似（DOS）的可扩展谱变点检测'
                           '（LAD 之可扩展版本）. ODD Workshop @ KDD, 2020；'
                           'Panzarasa P, Opsahl T, Carley K M. Patterns and dynamics of users’ behavior '
                           'and interaction: Network analysis of an online community. '
                           '<i>Journal of the American Society for Information Science and Technology</i>, '
                           '2009, 60(5): 911-932.（CollegeMsg 数据集来源，SNAP）', S["ref"]))
    story.append(Paragraph('[10] 内部资料：docs/查新核验-2026-09-14.md（DOSSIER 二次核验：P1/P6/P14/P15/'
                           'P18/P20/P21 CONFIRMED，P2 著录修正为 Lange 等 Calc. Var. PDE 2015；'
                           '反向查新确认 MDR 律、k=1 发现、vec-D 通道在经验律级新颖）；'
                           'notes/2026-09-15_P0l_SCPD精读与投稿路线.md（SCPD 精读、'
                           'G-REST/TRIP/RSDV 定位、venue 决策）.', S["ref"]))

    story.append(P('<a name="appA"/>附录 A　实验资产清单', "h1"))
    data = [["类别", "路径（项目根：streaming-fiedler-research/）"],
            ["计划与查新", "docs/研究计划v2-流式Fiedler.md；docs/查新报告.md；docs/NOVELTY_DOSSIER.md；docs/查新核验-2026-09-14.md"],
            ["代码基座", "experiments/trackers.py（跟踪器+插桩）；detectors.py；run_sim.py；dyngraph.py"],
            ["真实流装置", "experiments/real_data_noise.py（存 v<sub>2</sub>/λ<sub>3</sub>）；vec_drift.py；ou_detection.py"],
            ["LAD 基线/多数据集", "experiments/lad_baseline.py（--dataset email-eu|college-msg；含连通性门控变体）"],
            ["耦合标定", "experiments/track_quality.py；trackq_summary.py；gap_scaling.py；gap_regression.py"],
            ["受控实验", "experiments/synth_grid.py；synth_mdr.py"],
            ["谱隙强扫描", "experiments/synth_gap_scan.py；gap_scan_analysis.py（4 块层级 SBM 图族）"],
            ["校准依赖", "experiments/synth_tol_dep.py；tol_dep_analysis.py（13 校准组合 × 8 构型）"],
            ["断连 regime", "experiments/percomp_track.py；percomp_analysis.py（per-component，P0n）；"
             "trackvec_stream.py；trackvec_analysis.py（跟踪器轨迹，P0o）"],
            ["G-REST 对比", "experiments/grest_tracker.py（Alg.2 忠实复现，移位 T=cI−L）；"
             "grest_compare.py（P0h 装置流传模式同台）；runs/grest_compare.json；"
             "data/grest_2603.19439.pdf（全文）；notes/2026-09-15_P0p_GREST基线对比.md"],
            ["结果数据", "runs/*.npz（轨迹+双解+diag）；runs/lad_baseline*.json（双数据集三通道）；"
             "runs/synth_mdr.json；runs/synth_gap_scan*.json（135× 谱隙扫描）；"
             "runs/synth_tol_dep.json；runs/pc_*.npz；runs/tv_*.npz"],
            ["研究笔记", "notes/2026-09-14_{任务1-3, P0, P0b-P0i}_*.md；"
             "notes/2026-09-15_P0j_CollegeMsg跨数据集.md；notes/2026-09-15_P0k_谱隙强扫描.md；"
             "notes/2026-09-15_P0l_SCPD精读与投稿路线.md；notes/2026-09-15_P0m_tol依赖曲线.md；"
             "notes/2026-09-15_P0n_per-component谱跟踪.md；notes/2026-09-15_P0o_跟踪器轨迹vec-D.md；"
             "notes/2026-09-15_P0p_GREST基线对比.md"]]
    story.append(three_line(data, [3.4 * cm, 13 * cm]))

    doc = SimpleDocTemplate(OUT, pagesize=A4, title="流式 Fiedler 向量维护与谱结构变点检测：研究中期报告",
                            author="streaming-fiedler-research",
                            topMargin=2.2 * cm, bottomMargin=2.0 * cm,
                            leftMargin=2.2 * cm, rightMargin=2.2 * cm)
    doc.build(story, onFirstPage=first_page, onLaterPages=header_footer)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
