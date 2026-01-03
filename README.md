下面这套 JLaw scanner 的输出，本质是在帮你把流程拆成三层：
（1）强势股池 →（2）形态更干净的候选 →（3）当天已“过线+放量”的可执行突破。你每天收盘后就按这个顺序用就行。

1) 每个输出文件是干什么的
A. scan_outputs/metrics_all_YYYY-MM-DD.csv（你现在最重要的主表）

每只股票一行，把你需要做决策的“强度 + 形态 + 触发价 + 风险”都算好。

即使 candidates/breakouts 为空，你也能靠它手工挑出最值得盯的 10–30 只。

B. scan_outputs/candidates_YYYY-MM-DD.csv

在 metrics_all 的基础上，再加了一组更严格的“形态过滤”（near 52w high、pivot tight、vol dry-up、ATR 收缩、RS percentile 等）。

你设置得越严，这个文件越可能为空（这是正常的）。

C. scan_outputs/breakouts_YYYY-MM-DD.csv

从 candidates 里再筛 “今天已经过 trigger 且放量” 的（最接近“明天可以直接执行”的清单）。

为空也正常：说明今天没有满足你定义的“漂亮突破”。

D. data/jlaw_watchlist.txt

更干净的“强势形态 watchlist”（你可以把它作为后续 PP/PPP 的候选池，或者作为 TradingView/券商自选股列表）。

E. scan_outputs/jlaw_scanner_metrics_field_guide.csv

字段说明书（你忘了某列什么意思就查它）。

2) 你每天应该怎么用这份结果（收盘后 5–10 分钟流程）
Step 1：先在 metrics_all 里做“强度排序”

优先看这些列（从强到弱）：

rs_composite_pctile（综合相对强度百分位）

你的“领导股”核心指标。建议你先只看 ≥85 或 ≥90 的。

near_52w_gap_pct（离52周高点还差多少，越小越好）

例：0.05 = 距离 52w high 5%。一般你会喜欢 ≤0.20，更强可以 ≤0.10。

pivot_tight_range_pct（最近7天高低点区间/价格，越小越好）

形态“紧不紧”。常用阈值 ≤0.08（更苛刻 ≤0.06）。

这一层做完，你就会得到“强势股优先级列表”。

Step 2：判断它现在属于哪种“行动状态”

看这几列组合：

触发价（突破线）：breakout_trigger（在你脚本里也等于 trigger_price）

是否已经过线：

over_trigger_high：今天最高价是否过线（更宽松）

over_trigger_close：今天收盘是否过线（更严格）

放量是否足够：用

vol_today / vol50_avg（如果你 candidates/breakouts 里有 vol_surge_mult/vol_surge 就更直观）

你可以把状态理解成三类：

A) WATCH（还没到点）

over_trigger_high=False 且 (breakout_trigger - last_close)/breakout_trigger 还比较大

做法：不用交易，只需要“设提醒”。

B) NEAR TRIGGER（临门一脚，最值得盯）

自己算一下距离：

𝑑
𝑖
𝑠
𝑡
_
𝑡
𝑜
_
𝑡
𝑟
𝑖
𝑔
𝑔
𝑒
𝑟
=
𝑡
𝑟
𝑖
𝑔
𝑔
𝑒
𝑟
−
𝑙
𝑎
𝑠
𝑡
_
𝑐
𝑙
𝑜
𝑠
𝑒
𝑡
𝑟
𝑖
𝑔
𝑔
𝑒
𝑟
dist_to_trigger=
trigger
trigger−last_close
	​


若 ≤2%（你 yml 里 near_trigger_pct=0.02）就是“明天重点盯盘名单”。

做法：第二天开盘重点观察，等待突破触发。

C) BREAKOUT TODAY（今天已经过线）

over_trigger_high=True（或更严格用 close）

同时 vol_today/vol50_avg ≥ 1.5（你的 vol_breakout_mult=1.5）

做法：这就是你 breakouts 文件本应出现的票。明天如果不追高，仍在允许区间内可以考虑执行。

Step 3：把交易计划写成“可执行的三行”（触发 / 止损 / 仓位）

你表里已经给了关键字段：

入场触发：breakout_trigger
建议实际挂单用“略高于 trigger”的 buy stop（比如 +0.1%~+0.3%）避免假突破。

最大允许追价：max_entry_price

超过它（尤其是 gap up 直接越过），一般就不追。

止损参考：pivot_low7（你表里也叫 stop_suggest）

更稳健：止损可以略低于 pivot_low7（-0.2%~ -0.5% buffer），减少扫损。

风险高度：risk_pct = (trigger - stop) / trigger

这决定你“能不能用正常仓位做”。很多人会希望 ≤ 8%，再大就要减仓或放弃。

仓位计算（最实用）：
假设你账户每笔最多亏 R 美元（比如账户的 0.5% 或 1%），则

𝑠
ℎ
𝑎
𝑟
𝑒
𝑠
=
𝑅
𝑒
𝑛
𝑡
𝑟
𝑦
−
𝑠
𝑡
𝑜
𝑝
shares=
entry−stop
R
	​


其中 entry 可以用 trigger（或你实际 buy stop 价）。

3) 为什么你这次 Candidates/Breakouts 为空是“正常”的？

你现在的过滤条件挺严格（尤其叠加：RS percentile、near 52w、pivot tight、vol dry-up、ATR contraction、还要过线+放量）。
所以出现：

metrics_all 有 122/121 行（说明你 watchlist 里这些票有数据、指标算出来了）

但 candidates/breakouts 为空（说明 “没有任何票同时满足你定义的漂亮形态”）

这不是 bug，反而是“候选池更干净”的代价：很多天就是空仓/只观察。

4) 你真正“重点看”的字段清单（建议你记住这 10 个）

在 metrics_all 里按重要性：

rs_composite_pctile

near_52w_gap_pct

pivot_tight_range_pct

vol_contraction_score（越负越好，表示 ATR 收缩）

vol10_over_vol50（<1 越好，表示供给减少）

breakout_trigger（触发线）

max_entry_price（不追高红线）

pivot_low7（止损参考）

risk_pct（决定是否值得做、做多大）

over_trigger_high / over_trigger_close + vol_today/vol50_avg（决定今天是否已突破）

5) 你要怎么把它落到“明天要盯哪些票”

一个很实用的做法（不用改代码也能用）：

从 metrics_all 里筛：

rs_composite_pctile ≥ 85

near_52w_gap_pct ≤ 0.20

risk_pct ≤ 0.08（你自己的纪律）

然后手算（或加一列）dist_to_trigger，选 ≤2% 的票
→ 这就是“明日重点盯盘清单”。
